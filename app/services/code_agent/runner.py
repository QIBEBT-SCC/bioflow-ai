"""Isolated ACP runner and CODE proposal generation."""

import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

from acp import PROTOCOL_VERSION, Client, text_block
from acp.client.connection import ClientSideConnection
from acp.schema import ClientCapabilities, ElicitationCapabilities, ElicitationUrlCapabilities, Implementation
from acp.stdio import spawn_agent_process
from redis.asyncio import Redis

from app.core.config import get_settings
from app.schemas.code import CodeNodeType
from app.schemas.coding_agent import CodingAgentProvider
from app.services.code_agent import account as account_service
from app.services.code_agent import login as login_store
from app.services.code_agent import session as session_store
from app.services.code_agent.acp import CodeAgentClient
from app.services.code_agent.launch import build_agent_launch
from app.tasks.code import normalize_python_dependencies, normalize_r_dependencies
from app.utils.container import remove_container_process
from llm.utils.code_node_contract import SCRIPT_NAMES, code_agent_workspace_instructions

DEVICE_CODE_PATTERN = re.compile(r'\b[A-Z0-9]{4}-[A-Z0-9]{4}\b')


def _redis() -> Redis:
    return Redis.from_url(get_settings().persistent.cache.API_CACHE.unicode_string(), decode_responses=True)


def create_workspace(node_type: CodeNodeType) -> Path:
    """Create a session-only workspace with CODE authoring instructions."""
    workspace = Path(tempfile.mkdtemp(prefix='bioflow-code-agent-'))
    workspace.chmod(0o777)
    (workspace / 'output').mkdir(mode=0o777)
    agents = workspace / 'AGENTS.md'
    agents.write_text(code_agent_workspace_instructions(node_type), encoding='utf-8')
    agents.chmod(0o666)
    return workspace


def write_baseline(workspace: Path, node_type: CodeNodeType, source: str, dependencies: list[str]) -> None:
    """Replace only the files controlled by the CODE editor."""
    script = workspace / SCRIPT_NAMES[node_type]
    script.write_text(source, encoding='utf-8')
    script.chmod(0o666)
    dependency_path = workspace / 'dependencies.txt'
    if node_type is CodeNodeType.BASH:
        dependency_path.unlink(missing_ok=True)
    else:
        dependency_path.write_text('\n'.join(dependencies) + ('\n' if dependencies else ''), encoding='utf-8')
        dependency_path.chmod(0o666)


def _read_allowed_file(workspace: Path, filename: str) -> str:
    path = workspace / filename
    if path.is_symlink() or workspace.resolve() not in path.resolve(strict=True).parents:
        message = f'Agent replaced {filename} with an unsafe path'
        raise ValueError(message)
    return path.read_text(encoding='utf-8')


def read_candidate(workspace: Path, node_type: CodeNodeType) -> tuple[str, list[str], list[str]]:
    """Read and validate the allowed candidate files, ignoring everything else."""
    source = _read_allowed_file(workspace, SCRIPT_NAMES[node_type])
    if not source.strip():
        message = 'Agent produced an empty script'
        raise ValueError(message)
    dependencies: list[str] = []
    if node_type is not CodeNodeType.BASH:
        dependency_path = workspace / 'dependencies.txt'
        if dependency_path.exists():
            dependencies = _read_allowed_file(workspace, 'dependencies.txt').splitlines()
        dependencies = (
            normalize_python_dependencies(dependencies)
            if node_type is CodeNodeType.PYTHON
            else normalize_r_dependencies(dependencies)
        )
    allowed = {'AGENTS.md', SCRIPT_NAMES[node_type], 'output'}
    if node_type is not CodeNodeType.BASH:
        allowed.add('dependencies.txt')
    warnings = [f'Ignored extra file: {path.name}' for path in workspace.iterdir() if path.name not in allowed]
    return source, dependencies, warnings


def baseline_hash(node_type: CodeNodeType, source: str, dependencies: list[str]) -> str:
    """Hash the exact editor baseline associated with a proposal."""
    payload = json.dumps(
        {'node_type': node_type.value, 'source': source, 'dependencies': dependencies},
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def proposal_diff(
    node_type: CodeNodeType,
    before_source: str,
    after_source: str,
    before_dependencies: list[str],
    after_dependencies: list[str],
) -> str:
    """Render one unified diff for the whole turn."""
    filename = SCRIPT_NAMES[node_type]
    chunks = list(
        difflib.unified_diff(
            before_source.splitlines(keepends=True),
            after_source.splitlines(keepends=True),
            fromfile=f'a/{filename}',
            tofile=f'b/{filename}',
        )
    )
    if node_type is not CodeNodeType.BASH:
        chunks.extend(
            difflib.unified_diff(
                [f'{item}\n' for item in before_dependencies],
                [f'{item}\n' for item in after_dependencies],
                fromfile='a/dependencies.txt',
                tofile='b/dependencies.txt',
            )
        )
    return ''.join(chunks)


async def _drain_stderr(process: asyncio.subprocess.Process, callback: Any) -> None:
    if process.stderr is None:
        return
    while line := await process.stderr.readline():
        await callback(line.decode(errors='replace').rstrip())


@asynccontextmanager
async def start_acp(
    workspace: Path,
    auth_home: Path,
    name: str,
    on_update: Any,
    *,
    on_elicitation: Any = None,
    on_stderr: Any = None,
    provider: CodingAgentProvider = 'codex',
) -> AsyncGenerator[ClientSideConnection]:
    """Let the SDK own stdio and process lifecycle for the isolated adapter."""
    launch = build_agent_launch(name, workspace, auth_home, provider)
    async with spawn_agent_process(
        cast('Client', CodeAgentClient(on_update, on_elicitation)),
        *launch.argv,
        env={**os.environ, **launch.host_environment},
        transport_kwargs={'limit': 50 * 1024 * 1024},
        use_unstable_protocol=True,
    ) as (client, process):

        async def ignore_stderr(_line: str) -> None:
            return

        stderr_task = asyncio.create_task(_drain_stderr(process, on_stderr or ignore_stderr))
        try:
            response = await asyncio.wait_for(
                client.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(
                        terminal=False,
                        elicitation=ElicitationCapabilities(url=ElicitationUrlCapabilities())
                        if on_elicitation
                        else None,
                    ),
                    client_info=Implementation(name='bioflow-ai', title='BioFlowAI', version='0.1.0'),
                ),
                get_settings().code_agent.STARTUP_TIMEOUT_SECONDS,
            )
            if response.protocol_version != PROTOCOL_VERSION:
                _fail('ACP adapter does not support protocol v1')
            if on_elicitation and not any(
                method.id == 'chat-gpt-device-code' for method in (response.auth_methods or [])
            ):
                _fail('The configured codex-acp version does not support device-code login')
            yield client
        finally:
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task


def _normalize_update(update: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    payload = update.get('update')
    if not isinstance(payload, dict):
        return None
    kind = payload.get('sessionUpdate')
    if kind == 'session_info_update':
        return ('session.info', {'title': payload['title']}) if 'title' in payload else None
    if kind == 'usage_update':
        return 'session.usage', {key: payload[key] for key in ('used', 'size', 'cost') if key in payload}
    if kind == 'config_option_update':
        return 'config.updated', {'configOptions': payload.get('configOptions', [])}
    if kind == 'agent_message_chunk':
        content = payload.get('content')
        text = content.get('text') if isinstance(content, dict) else None
        return 'message.delta', {'text': text or ''}
    if kind == 'agent_thought_chunk':
        content = payload.get('content')
        text = content.get('text') if isinstance(content, dict) else None
        return 'thought.delta', {'message_id': payload.get('messageId'), 'text': text or ''}
    if kind in {'plan', 'plan_update', 'plan_removed'}:
        return 'plan.updated', payload
    if kind in {'tool_call', 'tool_call_update'}:
        raw_output = payload.get('rawOutput')
        if isinstance(raw_output, str):
            payload = {**payload, 'output': raw_output}
        elif isinstance(raw_output, dict):
            output = raw_output.get('output') or raw_output.get('stdout')
            if isinstance(output, str):
                payload = {**payload, 'output': output}
        return 'tool.updated', payload
    return None


def _device_code_from_message(message: str) -> str:
    match = DEVICE_CODE_PATTERN.search(message)
    return match.group(0) if match else ''


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _agent_error(exception: Exception, last_stderr: str | None = None) -> str:
    """Prefer the adapter's concrete stderr when the ACP transport only reports closure."""
    message = str(exception) or exception.__class__.__name__
    if last_stderr and message in {'Connection closed', 'Stream closed'}:
        return last_stderr
    return message


def _validate_device_url(value: object) -> str:
    if not isinstance(value, str):
        message = 'Codex returned no device verification URL'
        raise TypeError(message)
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or parsed.hostname != 'auth.openai.com':
        message = 'Codex returned an untrusted device verification URL'
        raise RuntimeError(message)
    return value


async def run_codex_login(login_id: uuid.UUID) -> None:
    """Complete one administrator-initiated ChatGPT device-code login."""
    redis = _redis()
    state = await redis.hgetall(login_store.login_key(login_id))  # pyrefly: ignore [not-async]
    if not state:
        await redis.aclose()
        return

    async def ignore_update(_update: dict[str, Any]) -> None:
        return

    async def publish_device_code(params: dict[str, Any]) -> dict[str, str]:
        if params.get('mode') != 'url':
            return {'action': 'cancel'}
        verification_url = _validate_device_url(params.get('url'))
        message = str(params.get('message') or '')
        await login_store.set_status(redis, login_id, 'waiting')
        await login_store.append_event(
            redis,
            login_id,
            'login.code',
            {
                'verification_url': verification_url,
                'user_code': _device_code_from_message(message),
                'message': message,
            },
        )
        return {'action': 'accept'}

    stack = AsyncExitStack()
    workspace: Path | None = None
    container_name = f'bioflow-acp-login-{login_id}'
    cancelled = False
    try:
        workspace = create_workspace(CodeNodeType.PYTHON)
        write_baseline(workspace, CodeNodeType.PYTHON, '# Codex login workspace\n', [])
        auth_home = account_service.prepare_auth_home_for_container()
        await login_store.append_event(redis, login_id, 'login.started', {})
        client = await stack.enter_async_context(
            start_acp(
                workspace,
                auth_home,
                container_name,
                ignore_update,
                on_elicitation=publish_device_code,
            )
        )
        login_task = asyncio.create_task(
            asyncio.wait_for(
                client.authenticate(method_id='chat-gpt-device-code'),
                get_settings().code_agent.LOGIN_TIMEOUT_SECONDS,
            )
        )
        while not login_task.done():
            if await login_store.cancel_requested(redis, login_id):
                cancelled = True
                login_task.cancel()
                with suppress(asyncio.CancelledError):
                    await login_task
                break
            await asyncio.sleep(1)
        if cancelled:
            account_service.remove_credentials()
            await login_store.append_event(redis, login_id, 'login.cancelled', {})
            await login_store.finish_login(redis, login_id, 'cancelled')
            return
        await login_task
        if not account_service.finalize_credentials():
            _fail('Codex device login completed without a reusable ChatGPT credential')
        await login_store.append_event(redis, login_id, 'login.completed', {})
        await login_store.finish_login(redis, login_id, 'completed')
    except Exception as exception:
        error = str(exception) or exception.__class__.__name__
        with suppress(Exception):
            account_service.remove_credentials()
        await login_store.append_event(redis, login_id, 'login.failed', {'message': error})
        await login_store.finish_login(redis, login_id, 'failed')
    finally:
        with suppress(Exception):
            await stack.aclose()
        await remove_container_process(container_name)
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
        await redis.aclose()


async def run_session(session_id: uuid.UUID) -> None:
    """Own one long-lived ACP subprocess and consume its Redis command stream."""
    redis = _redis()
    state = await redis.hgetall(session_store.session_key(session_id))  # pyrefly: ignore [not-async]
    if not state:
        await redis.aclose()
        return
    owner_id = int(state['owner_id'])
    node_type = CodeNodeType(state['node_type'])
    provider = cast('CodingAgentProvider', state.get('provider', 'codex'))
    if not account_service.credentials_configured(provider):
        await session_store.append_event(
            redis, session_id, 'turn.failed', {'message': f'{provider} credentials are not configured'}
        )
        await session_store.append_event(redis, session_id, 'session.closed', {})
        await session_store.release_session(redis, session_id, owner_id)
        await redis.aclose()
        return
    workspace: Path | None = None
    try:
        auth_home = account_service.prepare_auth_home_for_container(provider)
        workspace = create_workspace(node_type)
        write_baseline(workspace, node_type, '', [])
    except Exception as exception:
        await session_store.append_event(
            redis,
            session_id,
            'turn.failed',
            {'message': str(exception) or exception.__class__.__name__, 'recoverable': False},
        )
        await session_store.append_event(redis, session_id, 'session.closed', {})
        await session_store.release_session(redis, session_id, owner_id)
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
        await redis.aclose()
        return

    config_options: list[Any] = []

    async def publish_update(update: dict[str, Any]) -> None:
        event = _normalize_update(update)
        if event is not None:
            event_type, data = event
            await session_store.append_event(redis, session_id, event_type, data)

    last_stderr: str | None = None

    async def publish_stderr(line: str) -> None:
        nonlocal last_stderr
        if line.strip():
            last_stderr = line.strip()
        await session_store.append_event(redis, session_id, 'tool.updated', {'kind': 'terminal', 'output': line})

    stack = AsyncExitStack()
    command_cursor = '0-0'
    pending_snapshot: tuple[str, list[str]] | None = None
    try:
        client = await stack.enter_async_context(
            start_acp(
                workspace,
                auth_home,
                f'bioflow-acp-{session_id}',
                publish_update,
                on_stderr=publish_stderr,
                provider=provider,
            )
        )
        session = await asyncio.wait_for(
            client.new_session(cwd='/workspace', mcp_servers=[]),
            get_settings().code_agent.STARTUP_TIMEOUT_SECONDS,
        )
        acp_session_id = session.session_id
        config_options = session.config_options or []
        await session_store.append_event(
            redis,
            session_id,
            'config.updated',
            {
                'configOptions': [
                    option.model_dump(mode='json', by_alias=True, exclude_none=True) for option in config_options
                ]
            },
        )
        await session_store.set_status(redis, session_id, 'ready')
        await session_store.append_event(redis, session_id, 'session.ready', {'session_id': str(session_id)})

        close_requested = False
        while not close_requested:
            rows = await redis.xread(
                {session_store.command_key(session_id): command_cursor},
                count=1,
                block=get_settings().code_agent.HEARTBEAT_SECONDS * 1000,
            )
            if not rows:
                if not await redis.exists(session_store.session_key(session_id)):
                    break
                continue
            _stream, commands = rows[0]
            command_cursor, fields = commands[0]
            command = fields.get('command')
            data = json.loads(fields.get('data', '{}'))
            if command == 'close':
                close_requested = True
                continue
            if command == 'cancel':
                await client.cancel(session_id=acp_session_id)
                continue
            if command == 'config':
                try:
                    if pending_snapshot is not None:
                        _fail('Decide the pending proposal before changing configuration')
                    if not any(
                        option.id == data['config_id'] and option.category in {'model', 'thought_level'}
                        for option in config_options
                    ):
                        _fail('This session option is not available in the coding sidebar')
                    result = await asyncio.wait_for(
                        client.set_config_option(
                            session_id=acp_session_id, config_id=data['config_id'], value=data['value']
                        ),
                        get_settings().code_agent.STARTUP_TIMEOUT_SECONDS,
                    )
                    config_options = result.config_options
                    await session_store.append_event(
                        redis,
                        session_id,
                        'config.completed',
                        {
                            'configOptions': [
                                option.model_dump(mode='json', by_alias=True, exclude_none=True)
                                for option in config_options
                            ]
                        },
                    )
                except Exception as exception:
                    await session_store.append_event(redis, session_id, 'config.failed', {'message': str(exception)})
                continue
            if command == 'reject' and pending_snapshot is not None:
                write_baseline(workspace, node_type, *pending_snapshot)
                pending_snapshot = None
                continue
            if command == 'accept':
                pending_snapshot = None
                continue
            if command != 'turn':
                continue

            source = str(data.get('source', ''))
            dependencies = list(data.get('dependencies') or [])
            prompt = (
                code_agent_workspace_instructions(node_type) + '\n## Current Request\n\n' + str(data.get('prompt', ''))
            )
            pending_snapshot = (source, dependencies)
            write_baseline(workspace, node_type, source, dependencies)
            await session_store.set_status(redis, session_id, 'running')
            await session_store.append_event(redis, session_id, 'turn.started', {})
            prompt_task = asyncio.create_task(
                asyncio.wait_for(
                    client.prompt(session_id=acp_session_id, prompt=[text_block(prompt)]),
                    get_settings().code_agent.TURN_TIMEOUT_SECONDS,
                )
            )
            cancelled = False
            while not prompt_task.done():
                rows = await redis.xread(
                    {session_store.command_key(session_id): command_cursor},
                    count=1,
                    block=1000,
                )
                if not rows:
                    if not await redis.exists(session_store.session_key(session_id)):
                        cancelled = True
                        close_requested = True
                        await client.cancel(session_id=acp_session_id)
                    continue
                _stream, commands = rows[0]
                command_cursor, active_fields = commands[0]
                active_command = active_fields.get('command')
                if active_command == 'config':
                    await session_store.append_event(
                        redis, session_id, 'config.failed', {'message': 'Wait for the active turn to finish'}
                    )
                    continue
                if active_command in {'cancel', 'close'}:
                    cancelled = True
                    await client.cancel(session_id=acp_session_id)
                    if active_command == 'close':
                        close_requested = True
            try:
                await prompt_task
            except Exception as exception:
                event_type = 'turn.cancelled' if cancelled else 'turn.failed'
                await session_store.append_event(
                    redis,
                    session_id,
                    event_type,
                    {'message': str(exception), 'recoverable': not cancelled},
                )
                await session_store.set_status(redis, session_id, 'ready')
                write_baseline(workspace, node_type, source, dependencies)
                pending_snapshot = None
                continue
            if cancelled:
                await session_store.append_event(redis, session_id, 'turn.cancelled', {})
                await session_store.set_status(redis, session_id, 'ready')
                write_baseline(workspace, node_type, source, dependencies)
                pending_snapshot = None
                continue

            try:
                candidate_source, candidate_dependencies, warnings = read_candidate(workspace, node_type)
            except (OSError, UnicodeError, ValueError) as exception:
                await session_store.append_event(
                    redis,
                    session_id,
                    'turn.failed',
                    {'message': str(exception), 'recoverable': True},
                )
                await session_store.set_status(redis, session_id, 'ready')
                write_baseline(workspace, node_type, source, dependencies)
                pending_snapshot = None
                continue
            proposal_id = uuid.uuid4()
            proposal = {
                'id': str(proposal_id),
                'base_hash': baseline_hash(node_type, source, dependencies),
                'source': candidate_source,
                'dependencies': candidate_dependencies,
                'diff': proposal_diff(node_type, source, candidate_source, dependencies, candidate_dependencies),
                'warnings': warnings,
            }
            await session_store.save_proposal(redis, session_id, proposal_id, proposal)
            await session_store.append_event(redis, session_id, 'proposal.ready', proposal)
    except Exception as exception:
        if await redis.exists(session_store.session_key(session_id)):
            await session_store.set_status(redis, session_id, 'failed')
            await session_store.append_event(
                redis,
                session_id,
                'turn.failed',
                {'message': _agent_error(exception, last_stderr), 'recoverable': False},
            )
    finally:
        with suppress(Exception):
            await stack.aclose()
        await remove_container_process(f'bioflow-acp-{session_id}')
        shutil.rmtree(workspace, ignore_errors=True)
        await session_store.append_event(redis, session_id, 'session.closed', {})
        await session_store.release_session(redis, session_id, owner_id)
        await redis.aclose()
