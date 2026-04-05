#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 NOTEBOOK.ipynb [REPO_DIR]"
  exit 1
fi

NOTEBOOK_INPUT="$1"
REPO_DIR="${2:-.}"

cd "$REPO_DIR"

if [[ ! -f "$NOTEBOOK_INPUT" ]]; then
  echo "Error: notebook '$NOTEBOOK_INPUT' not found in '$REPO_DIR'"
  exit 1
fi

NOTEBOOK_BASENAME="$(basename "$NOTEBOOK_INPUT")"
NOTEBOOK_STEM="${NOTEBOOK_BASENAME%.ipynb}"

CLEAN_STEM="${NOTEBOOK_STEM#*_}"
if [[ "$CLEAN_STEM" == "$NOTEBOOK_STEM" ]]; then
  REPORT_TEX="exp_${NOTEBOOK_STEM}.tex"
else
  REPORT_TEX="exp_${CLEAN_STEM}.tex"
fi

PROMPT=$(cat <<EOF
Read \`$NOTEBOOK_BASENAME\` together with the associated package/source code and generate exactly these outputs:
1. \`$REPORT_TEX\`, a complete compilable LaTeX report;
2. a \`figures/\` directory containing only the key figures used in the report.

The report must be short (maximum 3 pages), self-contained, and written like a concise experimental note.

Required structure:
- \`\\subsection*{Setting}\`
- \`\\subsection*{Results}\`

Content requirements:
- In \`Setting\`, extract and state the concrete experimental setup as precisely as possible: target distribution or data/simulation setup, compared methods, architecture/backbone, training hyperparameters, evaluation protocol, and metrics.
- In \`Results\`, include only the most informative figures/tables from the notebook, with proper captions and labels, then briefly discuss the main findings, conclusions, and limitations.
- Reuse notebook outputs when possible; otherwise recreate the figures/tables faithfully from the notebook/code.
- Keep the writing compact, clear, and scientific.

Important constraints:
- Do not invent missing information.
- If some detail cannot be verified from the notebook or code, say so explicitly.
- Keep only useful figures in \`figures/\`.
- Deliver only \`$REPORT_TEX\` and \`figures/\`.
EOF
)

codex exec \
  --cd "$(pwd)" \
  --sandbox workspace-write \
  --ask-for-approval never \
  "$PROMPT"
