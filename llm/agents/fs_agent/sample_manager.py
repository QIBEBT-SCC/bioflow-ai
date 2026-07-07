from typing import TYPE_CHECKING, cast

from langchain.agents import create_agent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from llm.agents.fs_agent.analysis_reporter import AGENT_LLM
from llm.context import ProjectContext
from llm.models import get_agent_model
from llm.states import SampleState
from llm.tools.human import ask_human
from llm.tools.sample import (
    add_project_file,
    add_sample_file,
    add_samples,
    list_project_samples,
    run_command_tool,
    set_base_dir,
    write_conclusion,
)
from llm.utils.middleware import StatusBarMiddleware, TokenStatisticMiddleware
from llm.utils.middleware.final_check import CheckStateExistMiddleware

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

SYSTEM_PROMPT = """You are a bioinformatics project file system initialization agent.
Your task is to explore raw data paths provided by the user, identify the file structure, and register files into the system's two-level index.

## Two-level file

- **Sample-level** (`sample:<tag>`): Files belonging to individual samples, stored in `sample_files`.
- **Project-level** (`proj:<keyword>`): Shared files used across samples (e.g. BED annotation files), stored in `project_file_mappings`.

## Execution steps

1. **Check existing registry** — Always call `list_project_samples` first.
   - If the database already has samples:
     skip steps 1-3 entirely. Write the conclusion directly from the registry content via `write_conclusion`.
   - If no samples exist: proceed to steps 1-3 normally.
2. **Explore** — Call `set_base_dir`, then use `run_command` to map the directory structure.
   - `base_dir` is mounted as `/app/data` inside the container; use `/app/data` as the working path in all shell commands.
   - Drill into subdirectories as needed to build a complete file inventory.
   - If the path is inaccessible, call `ask_human` — this is the **only** situation to use it.
3. **Infer** — Determine how each file should be registered.
   - Files with one copy per sample → sample-level; files shared across all samples → project-level.
   - Choose a stable sample file `tag` from filename patterns; verify with `seqkit` or `head` when ambiguous.
   - Infer sequencing metadata (platform, technology) from filename patterns, directory names, or file headers; omit uncertain fields.
4. **Write to database** — Call `add_samples` → `add_sample_file` → `add_project_file` in order.
   - Group samples sharing the same metadata into a single `add_samples` call.
   - If the dataset spans multiple subdirectories with different data types or metadata, call `set_base_dir` again for each and repeat steps 1-3.
5. **Write conclusion** — Call `write_conclusion` with a structured Markdown summary (see template below).

## Conclusion template

```markdown
# Samples Info

### Sample metadata
- Sample count: N
- [Only include confirmed fields, e.g. platform, technology]

### Sample-level references
All samples contain the following tags:
- `sample:xxx` — [description]
- `sample:xxx` — [description]
[List custom tags if any]

### Project-level references
- `proj:xxx` — [description]
[Omit this section if none]

### Notes
[Special groupings, mixed data types, or anything downstream agents should be aware of]
```

Do not list individual file paths. Focus only on what downstream workflows can reference via `sample:` and `proj:` syntax.

## Notes
### Source of truth
- The database is the source of truth. Only re-scan the filesystem when the database is empty or the user
  explicitly states that local files have changed. Do not re-scan "to double-check" — that wastes time and tokens.

### Directory exploration
- `base_dir` is mounted as `/app/data` inside the container. Always use `/app/data` as the root path in shell commands — never use the host-side `base_dir` path directly.
- Do not inspect every file individually. Samples in the same batch typically share the same metadata; spot-check a few representative files to confirm, then apply uniformly.

### Sample naming

- Derive sample names from actual filenames or subdirectory names — do not invent arbitrary names.
- Sample names must be **unique** within the project; duplicates will cause registration to fail.

### Tag and keyword naming

The following are tag/keyword selections for some common scenarios:
| Pattern | tag |
|---|---|
| Paired-end R1 | `r1` |
| Paired-end R2 | `r2` |
| Single-end sequencing | `read` |
| Mass spectrometry | `spectrum` |
| Image file | `img` |
| Other files | `contig`, `bam`... |

- Format: lowercase letters and underscores only, max 50 characters.
- When a sample contains multiple files, append a descriptive **prefix**: `tumor_r1` / `normal_r1` / `light_img`, etc.
- Tags/Keywords must be **unique** within the project and semantically descriptive.
"""

MANAGER_LLM = get_agent_model()
metadata = cast('LLMMetadata', AGENT_LLM.metadata)


def build_create_sample_agent(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Create smaple agent."""
    if checkpointer:
        # pyrefly: ignore [no-matching-overload]
        return create_agent(
            MANAGER_LLM,
            [
                set_base_dir,
                run_command_tool,
                list_project_samples,
                add_samples,
                add_sample_file,
                add_project_file,
                write_conclusion,
                ask_human,
            ],
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                StatusBarMiddleware(),
                CheckStateExistMiddleware('summary', 'write_conclusion'),
                TokenStatisticMiddleware('sample_manager', **metadata),
            ],
            state_schema=SampleState,
            context_schema=ProjectContext,
            checkpointer=checkpointer,
        )
    # pyrefly: ignore [no-matching-overload]
    return create_agent(
        MANAGER_LLM,
        [
            set_base_dir,
            run_command_tool,
            list_project_samples,
            add_samples,
            add_sample_file,
            add_project_file,
            write_conclusion,
            ask_human,
        ],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            StatusBarMiddleware(),
            CheckStateExistMiddleware('summary', 'write_conclusion'),
            TokenStatisticMiddleware('sample_manager', **metadata),
        ],
        state_schema=SampleState,
        context_schema=ProjectContext,
    )


create_sample_agent = build_create_sample_agent()
