#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 NOTEBOOK.ipynb [REPO_DIR]" >&2
}

die() {
  echo "Error: $*" >&2
  exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

readonly NOTEBOOK_INPUT="$1"
readonly REPO_DIR="${2:-.}"

[[ -d "$REPO_DIR" ]] || die "repo directory '$REPO_DIR' not found."

cd "$REPO_DIR"

[[ -f "$NOTEBOOK_INPUT" ]] || die "notebook '$NOTEBOOK_INPUT' not found in '$(pwd)'."
[[ "$NOTEBOOK_INPUT" == *.ipynb ]] || die "expected a .ipynb file, got '$NOTEBOOK_INPUT'."
command -v codex >/dev/null 2>&1 || die "'codex' command not found."

readonly NOTEBOOK_BASENAME="$(basename "$NOTEBOOK_INPUT")"
readonly NOTEBOOK_STEM="${NOTEBOOK_BASENAME%.ipynb}"

readonly CLEAN_STEM="${NOTEBOOK_STEM#*_}"
if [[ "$CLEAN_STEM" == "$NOTEBOOK_STEM" ]]; then
  readonly REPORT_TEX="exp_${NOTEBOOK_STEM}.tex"
else
  readonly REPORT_TEX="exp_${CLEAN_STEM}.tex"
fi

readonly PROMPT_TEMPLATE=$(cat <<'EOF'
Read `__NOTEBOOK_BASENAME__` and the associated source/package files, then generate exactly these outputs and nothing else:

1. `__REPORT_TEX__`: a complete, standalone, compilable LaTeX report.
2. `figures/`: a directory containing only the figure files used in the report.

Global objective
Produce a very short, visually light, self-contained experimental note that captures the notebook's main empirical content with minimal text and maximal use of informative figures/tables.

Hard constraints
- Maximum length: 5 pages total, including figures, tables, and references.
- Deliver only `__REPORT_TEX__` and `figures/`.
- Do not create any extra files.
- Do not invent, guess, or fill in missing information.
- If a detail cannot be verified from the notebook or source files, state explicitly that it could not be verified, or omit it if it is not necessary.
- Do not mention “this notebook”, “the code”, “the repository”, or similar.
- Write in concise third-person scientific style, as if reporting a completed experiment.
- Avoid useless prose, repetition, and verbatim restatement of notebook text.

Writing requirements
- Write as little as possible while remaining precise.
- Prefer structure, whitespace, short paragraphs, and figures/tables over long prose.
- Keep the page visually light: short sentences, compact paragraphs, sparse commentary.
- Do not include generic background, motivation, or textbook explanations.
- Do not describe implementation details unless they are experimentally necessary and verifiable.

Required structure
- `\subsection*{Setting}`
- `\subsection*{Results}`

Content requirements

`Setting`
Report only the verified experimental setup, as concretely as possible. Include, when available:
- target distribution, dataset, or simulation setup;
- compared methods or baselines;
- architecture / backbone;
- training hyperparameters and optimization details;
- evaluation protocol;
- reported metrics.

Keep this section extremely short. Prefer dense factual reporting over explanation.

`Results`
This section should carry most of the report.
- Reuse notebook outputs whenever possible.
- Otherwise recreate figures/tables faithfully from the notebook and source files.
- Prefer including many informative figures rather than summarizing them in text, as long as the 5-page limit is respected.
- Use the notebook's strongest figures first: final comparison plots, ablations, quantitative summaries, representative qualitative outputs, and compact tables.
- Exclude only figures that are redundant, low-information, purely diagnostic, or not needed to support the main findings.
- Every figure/table must have a short caption and a label.
- Arrange content so the results are easy to scan visually.
- Add only minimal commentary: one or two short sentences per group of figures/tables, limited to what is directly supported by the displayed evidence.

Figure policy
- `figures/` must contain only assets actually referenced in `__REPORT_TEX__`.
- Keep as many non-redundant, informative notebook figures as possible within the page limit.
- Prefer multi-panel figures or compact layouts when this helps include more useful results without clutter.
- Do not keep duplicate or near-duplicate visuals.
- Do not include raw screenshots unless they are the only faithful way to preserve a notebook result.

Style policy
- Minimal wording.
- No decorative phrasing.
- No long verbatim extraction from notebook markdown/text.
- No narrative filler.
- No unsupported interpretation.
- Let the figures and tables do most of the work.

Final output contract
Return exactly:
- `__REPORT_TEX__`
- `figures/`
and ensure `__REPORT_TEX__` compiles and references only files present in `figures/`.
EOF
)

PROMPT="${PROMPT_TEMPLATE//__NOTEBOOK_BASENAME__/$NOTEBOOK_BASENAME}"
PROMPT="${PROMPT//__REPORT_TEX__/$REPORT_TEX}"

codex exec \
  --cd "$(pwd)" \
  --sandbox workspace-write \
  "$PROMPT"
