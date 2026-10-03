import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

POSTGRES_URL = os.environ.get('BIOFLOW_MIGRATION_POSTGRES_URL')
REDIS_URL = os.environ.get('BIOFLOW_MIGRATION_REDIS_URL')


@pytest.mark.integration
@pytest.mark.skipif(not POSTGRES_URL or not REDIS_URL, reason='Requires dedicated temporary PostgreSQL and Redis')
def test_sql_initialization_auth_and_redis_checkpoints(tmp_path: Path) -> None:
    # Run separately so settings, engines, and cached module globals cannot
    # accidentally inherit normal application connections from another test.
    script = dedent("""
        import asyncio
        import os
        from decimal import Decimal
        from urllib.parse import urlparse
        from pydantic import PostgresDsn, RedisDsn, SecretStr
        from sqlalchemy import inspect
        from sqlalchemy.orm import Session
        from bioflow_ai.core import config

        pg_url = os.environ['BIOFLOW_MIGRATION_POSTGRES_URL']
        redis_url = os.environ['BIOFLOW_MIGRATION_REDIS_URL']
        pg = urlparse(pg_url)
        redis = urlparse(redis_url)
        assert pg.hostname == redis.hostname == '127.0.0.1'
        assert pg.path == '/bioflow_layout_test'
        assert pg.port != 5432 and redis.port != 6379
        settings = config.Settings()
        settings.persistent.database.DB_URL = PostgresDsn(pg_url)
        settings.persistent.cache.API_CACHE = RedisDsn(redis_url)
        settings.persistent.cache.LLM_CACHE = RedisDsn(redis_url)
        settings.persistent.cache.MONITOR_CACHE = RedisDsn(redis_url)
        settings.persistent.cache.NOTIFICATION_CACHE = RedisDsn(redis_url)
        config.get_settings = lambda: settings

        from bioflow_ai.infrastructure.database import engine
        from bioflow_ai.models import Base
        # The fresh container already ran the unchanged presets/sql mount.
        assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())
        Base.metadata.create_all(engine)
        from bioflow_ai.services.initialization.database import init_admin
        from bioflow_ai.schemas.user import UserCreate
        from bioflow_ai.services.identity.user import authenticate_user
        from bioflow_ai.services.identity.auth import create_session, get_session, delete_session
        from bioflow_ai.infrastructure.redis.client import setup_cache_redis, close_cache_regis

        from bioflow_ai.llm import models
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        FakeListChatModel.with_structured_output = lambda self, *args, **kwargs: self
        FakeListChatModel.bind_tools = lambda self, *args, **kwargs: self
        def fake_model():
            model = FakeListChatModel(responses=['offline'])
            model.metadata = dict(model_id=1, setting_key='coding', input_price=Decimal(0),
                                  output_price=Decimal(0), cache_read_price=Decimal(0))
            return model
        for name in dir(models):
            if name.startswith('get_') and name.endswith('_model'):
                setattr(models, name, fake_model)
        from bioflow_ai.tasks.agent_run import _checkpoint_serializer
        from bioflow_ai.schemas.tool_config import ImageConfig
        from langgraph.checkpoint.base import empty_checkpoint
        from langgraph.checkpoint.redis import AsyncRedisSaver

        async def run():
            credentials = UserCreate(username='layout-test', email='layout@example.test',
                                     password=SecretStr('migration-test-password'))
            assert (await init_admin(credentials)).created
            assert not (await init_admin()).created
            with Session(engine) as session:
                user = authenticate_user(session, credentials.email, credentials.password.get_secret_value())
                assert user is not None
                user_id = user.id
                assert authenticate_user(session, credentials.email, 'wrong password') is None
            cache = await setup_cache_redis()
            try:
                issued = await create_session(cache, user_id)
                session = await get_session(cache, issued.session_id)
                assert session is not None and session.record.user_id == user_id
                await delete_session(cache, issued.session_id)
                assert await get_session(cache, issued.session_id) is None
                async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
                    saver.serde = _checkpoint_serializer()
                    checkpoint = empty_checkpoint()
                    image = ImageConfig(repository='layout-test')
                    checkpoint['channel_values'] = {'image': image}
                    checkpoint['channel_versions'] = {'image': 1}
                    saved_config = await saver.aput(
                        {'configurable': {'thread_id': 'layout-test', 'checkpoint_ns': ''}},
                        checkpoint, {}, {'image': 1}
                    )
                    loaded = await saver.aget(saved_config)
                    assert loaded is not None
                    assert loaded['channel_values']['image'] == image
            finally:
                await close_cache_regis(cache)
                engine.dispose()
        asyncio.run(run())
        print('SQL presets, admin persistence/login, Redis sessions and checkpoint round trip passed')
    """)
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
