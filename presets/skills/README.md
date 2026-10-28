# Agent Skills

Skills are assigned by directory:

```text
presets/skills/
└── <agent-name>/
    └── <skill-name>/
        ├── SKILL.md
        ├── references/  # optional
        ├── scripts/     # optional
        └── assets/      # optional
```

`SkillMiddleware('<agent-name>')` automatically discovers every immediate skill directory belonging to that agent.
If two agents need the same skill, keep a copy under each agent directory.

Each `SKILL.md` follows the Agent Skills format:

```markdown
---
name: skill-name
description: What this skill does and when the agent should use it.
---

# Skill instructions

Detailed workflow instructions go here. Link larger supporting material with paths relative to the skill directory,
for example `references/DETAILS.md`.
```

The `name` must match its directory and contain only lowercase letters, numbers, and hyphens. At startup, only skill
names and descriptions enter the model context. The agent loads `SKILL.md` with `load_skill`, then reads referenced
text resources individually with `read_skill_resource`.

The CODE agent's language instructions live in `presets/code/{bash,python,r}/AGENTS.md`. Initialization copies the
selected language's template into its workspace, and the agent reads that file before editing code.
The CODE agent keeps `SkillMiddleware('coding_agent')` enabled; `presets/skills/coding_agent/` is currently empty and
can hold programming skills as they are added.
