---
name: writing-bash-code
description: Create or edit script.sh for a BioFlowAI code_bash CODE definition using its available Linux tools, exact getopts inputs, and no runtime dependencies.
---

# Write a Bash CODE definition

## Authoring Files and Checks

- When editing an authoring workspace, deliver the script in `/workspace/script.sh`.
- Do not create `dependencies.txt`; Bash CODE definitions cannot declare runtime dependencies.
- Use `bash -n script.sh` in the biokit-utils inspection container to check syntax. A syntax check does not establish
  functional correctness; report any additional checks that actually ran.

## Target Runtime

- The script runs with `/usr/bin/bash`.
- Its current working directory is a newly created writable output directory dedicated to this node invocation.
- Upstream inputs are passed as named command-line options from `bind_param` nodes.
- Standard Linux commands plus `fd`, `rg`, `jq`, `yq`, and `mlr` are available.
- Do not assume Python, R, third-party packages, bioinformatics programs, package installation, or network access.

## Request Eligibility

- Before generating code, determine whether the entire request can be implemented in the runtime described above.
- Reject the request when it requires Python, R, a bioinformatics program, any other unavailable executable or
  third-party package, package installation, or network access.
- Do not replace a requested bioinformatics tool or Python/R program with an improvised Bash implementation.

## Script Requirements

- Start with `#!/usr/bin/env bash` and `set -euo pipefail`.
- Parse every declared input using `getopts`; never rely on argument order.
- Use exactly the short options provided by the authoritative input contract.
- Quote every path and validate all required options and expected file or directory types.
- Write results to the current working directory using deterministic plain file names.
- Never modify or write beside upstream inputs.
- Send diagnostics to stderr and exit non-zero for invalid input.
- Avoid writing large data to stdout; redirect results directly to named files.
- Do not invoke Python or R.
- Do not use bioinformatics software; such operations belong in tool nodes.
