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

readonly NOTEBOOK_REF="$NOTEBOOK_INPUT"
readonly NOTEBOOK_BASENAME="$(basename "$NOTEBOOK_INPUT")"
readonly STYLE_CONTENT=$(cat <<'EOF_STYLE'
% partial rewrite of the LaTeX2e package for submissions to the
% Conference on Neural Information Processing Systems (NeurIPS):
%
% - uses more LaTeX conventions
% - line numbers at submission time replaced with aligned numbers from
%   lineno package
% - \nipsfinalcopy replaced with [final] package option
% - automatically loads times package for authors
% - loads natbib automatically; this can be suppressed with the
%   [nonatbib] package option
% - adds foot line to first page identifying the conference
% - adds preprint option for submission to e.g. arXiv
% - conference acronym modified
% - update foot line to display the track name
%
% Roman Garnett (garnett@wustl.edu) and the many authors of
% nips15submit_e.sty, including MK and drstrip@sandia
%
% last revision: January 2026

\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{neurips_2026}[2026-01-29 NeurIPS 2026 submission/camera-ready style file]

% declare final option, which creates camera-ready copy
\newif\if@neuripsfinal\@neuripsfinalfalse
\DeclareOption{final}{
  \@neuripsfinaltrue
  \@anonymousfalse
}

% declare nonatbib option, which does not load natbib in case of
% package clash (users can pass options to natbib via
% \PassOptionsToPackage)
\newif\if@natbib\@natbibtrue
\DeclareOption{nonatbib}{
  \@natbibfalse
}

% declare preprint option, which creates a preprint version ready for
% upload to, e.g., arXiv
\newif\if@preprint\@preprintfalse
\DeclareOption{preprint}{
  \@preprinttrue
  \@anonymousfalse
}

% determine the track of the paper in camera-ready mode
\newif\if@main\@maintrue
\DeclareOption{main}{
  \@maintrue
  \newcommand{\@trackname}{\@neuripsordinal\ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear).}
}
\newif\if@position\@positionfalse
\DeclareOption{position}{
  \@positiontrue
  \newcommand{\@trackname}{\@neuripsordinal\ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Position Paper Track.}
}
\newif\if@eandd\@eanddfalse
\DeclareOption{eandd}{
  \@eanddtrue
\if@neuripsfinal\@anonymousfalse\else\if@preprint\@anonymousfalse\else\@anonymoustrue\fi\fi
  \newcommand{\@trackname}{\@neuripsordinal\ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Track on Evaluations and Datasets.}
}
\newif\if@creativeai\@creativeaifalse
\DeclareOption{creativeai}{
  \@creativeaitrue
  \@anonymousfalse
  \newcommand{\@trackname}{\@neuripsordinal\ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Creative AI Track.}
}

% For anonymous or non-anonymous
\newif\if@anonymous\@anonymoustrue

% For workshop papers
\newcommand{\@workshoptitle}{}
\newcommand{\workshoptitle}[1]{\renewcommand{\@workshoptitle}{#1}}

\newif\if@workshop\@workshopfalse
\DeclareOption{sglblindworkshop}{
  \@workshoptrue
  \@anonymousfalse
  \newcommand{\@trackname}{\@neuripsordinal\ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Workshop: \@workshoptitle.}
}
\DeclareOption{dblblindworkshop}{
  \@workshoptrue
  \newcommand{\@trackname}{\@neuripsordinal\ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Workshop: \@workshoptitle.}
}
\DeclareOption{nonanonymous}{
  \@anonymousfalse
}

\ProcessOptions\relax

% fonts
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}

% change this every year for notice string at bottom
\newcommand{\@neuripsordinal}{40th}
\newcommand{\@neuripsyear}{2026}
\newcommand{\@neuripslocation}{Sydney}

% acknowledgments
\usepackage{environ}
\newcommand{\acksection}{\section*{Acknowledgments and Disclosure of Funding}}
\NewEnviron{ack}{%
  \acksection
  \BODY
}


% load natbib unless told otherwise
\if@natbib
  \RequirePackage{natbib}
\fi





% set page geometry
\usepackage[verbose=true,letterpaper]{geometry}
\AtBeginDocument{
  \newgeometry{
    textheight=9in,
    textwidth=5.5in,
    top=1in,
    headheight=12pt,
    headsep=25pt,
    footskip=30pt
  }
  \@ifpackageloaded{fullpage}
    {\PackageWarning{neurips_2026}{fullpage package not allowed! Overwriting formatting.}}
    {}
}

\widowpenalty=10000
\clubpenalty=10000
\flushbottom
\sloppy


% font sizes with reduced leading
\renewcommand{\normalsize}{%
  \@setfontsize\normalsize\@xpt\@xipt
  \abovedisplayskip      7\p@ \@plus 2\p@ \@minus 5\p@
  \abovedisplayshortskip \z@ \@plus 3\p@
  \belowdisplayskip      \abovedisplayskip
  \belowdisplayshortskip 4\p@ \@plus 3\p@ \@minus 3\p@
}
\normalsize
\renewcommand{\small}{%
  \@setfontsize\small\@ixpt\@xpt
  \abovedisplayskip      6\p@ \@plus 1.5\p@ \@minus 4\p@
  \abovedisplayshortskip \z@  \@plus 2\p@
  \belowdisplayskip      \abovedisplayskip
  \belowdisplayshortskip 3\p@ \@plus 2\p@   \@minus 2\p@
}
\renewcommand{\footnotesize}{\@setfontsize\footnotesize\@ixpt\@xpt}
\renewcommand{\scriptsize}{\@setfontsize\scriptsize\@viipt\@viiipt}
\renewcommand{\tiny}{\@setfontsize\tiny\@vipt\@viipt}
\renewcommand{\large}{\@setfontsize\large\@xiipt{14}}
\renewcommand{\Large}{\@setfontsize\Large\@xivpt{16}}
\renewcommand{\LARGE}{\@setfontsize\LARGE\@xviipt{20}}
\renewcommand{\huge}{\@setfontsize\huge\@xxpt{23}}
\renewcommand{\Huge}{\@setfontsize\Huge\@xxvpt{28}}


% Force \tiny to be no smaller than 6pt
\renewcommand{\tiny}{\fontsize{6pt}{7pt}\selectfont}

% Force \scriptsize to be no smaller than 7pt
\renewcommand{\scriptsize}{\fontsize{7pt}{8pt}\selectfont}

% Force \footnotesize to be no smaller than 8pt
\renewcommand{\footnotesize}{\fontsize{8pt}{9.5pt}\selectfont}

% sections with less space
\providecommand{\section}{}
\renewcommand{\section}{%
  \@startsection{section}{1}{\z@}%
                {-2.0ex \@plus -0.5ex \@minus -0.2ex}%
                { 1.5ex \@plus  0.3ex \@minus  0.2ex}%
                {\large\bf\raggedright}%
}
\providecommand{\subsection}{}
\renewcommand{\subsection}{%
  \@startsection{subsection}{2}{\z@}%
                {-1.8ex \@plus -0.5ex \@minus -0.2ex}%
                { 0.8ex \@plus  0.2ex}%
                {\normalsize\bf\raggedright}%
}
\providecommand{\subsubsection}{}
\renewcommand{\subsubsection}{%
  \@startsection{subsubsection}{3}{\z@}%
                {-1.5ex \@plus -0.5ex \@minus -0.2ex}%
                { 0.5ex \@plus  0.2ex}%
                {\normalsize\bf\raggedright}%
}
\providecommand{\paragraph}{}
\renewcommand{\paragraph}{%
  \@startsection{paragraph}{4}{\z@}%
                {1.5ex \@plus 0.5ex \@minus 0.2ex}%
                {-1em}%
                {\normalsize\bf}%
}
\providecommand{\subparagraph}{}
\renewcommand{\subparagraph}{%
  \@startsection{subparagraph}{5}{\z@}%
                {1.5ex \@plus 0.5ex \@minus 0.2ex}%
                {-1em}%
                {\normalsize\bf}%
}
\providecommand{\subsubsubsection}{}
\renewcommand{\subsubsubsection}{%
  \vskip5pt{\noindent\normalsize\rm\raggedright}%
}

% float placement
\renewcommand{\topfraction      }{0.85}
\renewcommand{\bottomfraction   }{0.4}
\renewcommand{\textfraction     }{0.1}
\renewcommand{\floatpagefraction}{0.7}

\newlength{\@neuripsabovecaptionskip}\setlength{\@neuripsabovecaptionskip}{7\p@}
\newlength{\@neuripsbelowcaptionskip}\setlength{\@neuripsbelowcaptionskip}{\z@}

\setlength{\abovecaptionskip}{\@neuripsabovecaptionskip}
\setlength{\belowcaptionskip}{\@neuripsbelowcaptionskip}

% swap above/belowcaptionskip lengths for tables
\renewenvironment{table}
  {\setlength{\abovecaptionskip}{\@neuripsbelowcaptionskip}%
   \setlength{\belowcaptionskip}{\@neuripsabovecaptionskip}%
   \@float{table}}
  {\end@float}

% footnote formatting
\setlength{\footnotesep }{6.65\p@}
\setlength{\skip\footins}{9\p@ \@plus 4\p@ \@minus 2\p@}
\renewcommand{\footnoterule}{\kern-3\p@ \hrule width 12pc \kern 2.6\p@}
\setcounter{footnote}{0}

% paragraph formatting
\setlength{\parindent}{\z@}
\setlength{\parskip  }{5.5\p@}

% list formatting
\setlength{\topsep       }{4\p@ \@plus 1\p@   \@minus 2\p@}
\setlength{\partopsep    }{1\p@ \@plus 0.5\p@ \@minus 0.5\p@}
\setlength{\itemsep      }{2\p@ \@plus 1\p@   \@minus 0.5\p@}
\setlength{\parsep       }{2\p@ \@plus 1\p@   \@minus 0.5\p@}
\setlength{\leftmargin   }{3pc}
\setlength{\leftmargini  }{\leftmargin}
\setlength{\leftmarginii }{2em}
\setlength{\leftmarginiii}{1.5em}
\setlength{\leftmarginiv }{1.0em}
\setlength{\leftmarginv  }{0.5em}
\def\@listi  {\leftmargin\leftmargini}
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep  2\p@ \@plus 1\p@    \@minus 0.5\p@
              \parsep  1\p@ \@plus 0.5\p@ \@minus 0.5\p@
              \itemsep \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    1\p@ \@plus 0.5\p@ \@minus 0.5\p@
              \parsep    \z@
              \partopsep 0.5\p@ \@plus 0\p@ \@minus 0.5\p@
              \itemsep \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

% create title
\providecommand{\maketitle}{}
\renewcommand{\maketitle}{%
  \par
  \begingroup
    \renewcommand{\thefootnote}{\fnsymbol{footnote}}
    % for perfect author name centering
    \renewcommand{\@makefnmark}{\hbox to \z@{$^{\@thefnmark}$\hss}}
    % The footnote-mark was overlapping the footnote-text,
    % added the following to fix this problem               (MK)
    \long\def\@makefntext##1{%
      \parindent 1em\noindent
      \hbox to 1.8em{\hss $\m@th ^{\@thefnmark}$}##1
    }
    \thispagestyle{empty}
    \@maketitle
    \@thanks
    \@notice
  \endgroup
  \let\maketitle\relax
  \let\thanks\relax
}

% rules for title box at top of first page
\newcommand{\@toptitlebar}{
  \hrule height 4\p@
  \vskip 0.25in
  \vskip -\parskip%
}
\newcommand{\@bottomtitlebar}{
  \vskip 0.29in
  \vskip -\parskip
  \hrule height 1\p@
  \vskip 0.09in%
}

% create title (includes both anonymized and non-anonymized versions)
\providecommand{\@maketitle}{}
\renewcommand{\@maketitle}{%
  \vbox{%
    \hsize\textwidth
    \linewidth\hsize
    \vskip 0.1in
    \@toptitlebar
    \centering
    {\LARGE\bf \@title\par}
    \@bottomtitlebar
    \if@anonymous
      \begin{tabular}[t]{c}\bf\rule{\z@}{24\p@}
        Anonymous Author(s) \\
        Affiliation \\
        Address \\
        \texttt{email} \\
      \end{tabular}%
    \else
      \def\And{%
        \end{tabular}\hfil\linebreak[0]\hfil%
        \begin{tabular}[t]{c}\bf\rule{\z@}{24\p@}\ignorespaces%
      }
      \def\AND{%
        \end{tabular}\hfil\linebreak[4]\hfil%
        \begin{tabular}[t]{c}\bf\rule{\z@}{24\p@}\ignorespaces%
      }
      \begin{tabular}[t]{c}\bf\rule{\z@}{24\p@}\@author\end{tabular}%
    \fi
    \vskip 0.3in \@minus 0.1in
  }
}

% add conference notice to bottom of first page
\newcommand{\ftype@noticebox}{8}
\newcommand{\@notice}{%
  % give a bit of extra room back to authors on first page
  \enlargethispage{2\baselineskip}%
  \@float{noticebox}[b]%
    \footnotesize\@noticestring%
  \end@float%
}

% abstract styling
\renewenvironment{abstract}%
{%
  \vskip 0.075in%
  \centerline%
  {\large\bf Abstract}%
  \vspace{0.5ex}%
  \begin{quote}%
}
{
  \par%
  \end{quote}%
  \vskip 1ex%
}

% For the paper checklist
\newcommand{\answerYes}[1][]{\textcolor{blue}{[Yes]#1}}
\newcommand{\answerNo}[1][]{\textcolor{orange}{[No]#1}}
\newcommand{\answerNA}[1][]{\textcolor{gray}{[N/A]#1}}
\newcommand{\answerTODO}[1][]{\textcolor{red}{\bf [TODO]}}
\newcommand{\justificationTODO}[1][]{\textcolor{red}{\bf [TODO]}}

% handle tweaks for camera-ready copy vs. submission copy
\if@preprint
  \newcommand{\@noticestring}{%
    Preprint.%
  }
\else
  \if@neuripsfinal
    \newcommand{\@noticestring}{
      \@trackname
    }
  \else
    \newcommand{\@noticestring}{%
      Submitted to \@neuripsordinal\/ Conference on Neural Information Processing Systems (NeurIPS \@neuripsyear). Do not distribute.%
    }

    % hide the acknowledgements
    \NewEnviron{hide}{}
    \let\ack\hide
    \let\endack\endhide

    % line numbers for submission
    \RequirePackage{lineno}
    \linenumbers

    % fix incompatibilities between lineno and amsmath, if required, by
    % transparently wrapping linenomath environments around amsmath
    % environments
    \AtBeginDocument{%
      \@ifpackageloaded{amsmath}{%
        \newcommand*\patchAmsMathEnvironmentForLineno[1]{%
          \expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
          \expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
          \renewenvironment{#1}%
                          {\linenomath\csname old#1\endcsname}%
                          {\csname oldend#1\endcsname\endlinenomath}%
        }%
        \newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{%
          \patchAmsMathEnvironmentForLineno{#1}%
          \patchAmsMathEnvironmentForLineno{#1*}%
        }%
        \patchBothAmsMathEnvironmentsForLineno{equation}%
        \patchBothAmsMathEnvironmentsForLineno{align}%
        \patchBothAmsMathEnvironmentsForLineno{flalign}%
        \patchBothAmsMathEnvironmentsForLineno{alignat}%
        \patchBothAmsMathEnvironmentsForLineno{gather}%
        \patchBothAmsMathEnvironmentsForLineno{multline}%
      }
      {}
    }
  \fi
\fi


\endinput
EOF_STYLE
)

readonly PROMPT_TEMPLATE=$(cat <<'EOF'
Read `__NOTEBOOK_BASENAME__` and the associated source/package files, then generate exactly one output directory and nothing else:

`report/`, containing exactly:
- `main.tex`: a complete, standalone, compilable LaTeX report;
- `bibliography.bib`: a BibTeX file containing only the references cited in the document;
- `neurips_2026.sty`: a copy of the style file, copied from the source directory;
- `figures/`: a directory containing only the figure files used in the report.

Global objective
Produce a short, readable experimental note in NeurIPS style. The goal is good formatting and easy reading, not imitation of a polished conference submission. The document should summarize the main empirical content in a compact but readable way, with strong use of informative figures and tables, plus a compact appendix with formal definitions needed to make the document self-contained.

Scope and execution note
- The source material may be long. That is acceptable.
- Read the full relevant material before writing.
- Do not rush to produce outputs after partial inspection.
- It is better to spend time extracting the correct experimental content than to produce an incomplete or shallow summary.

Style and template requirements
- Use `neurips_2026.sty`, and place a copy of it inside `report/`.
- Format the document as a NeurIPS paper purely for readability.
- Use a generic, descriptive title.
- Include a short abstract.
- Write in concise third-person scientific style.
- Do not try to mimic the tone or rhetorical structure of a polished conference submission.
- Do not add paper-like filler, hype, novelty framing, or broad motivational prose.

Hard constraints
- Maximum length of the main body: 10 pages total, including title, abstract, all main sections, figures, and tables.
- The references do not count toward the 10-page limit.
- The appendix does not count toward the 10-page limit.
- References must start on a new page immediately after the `Analysis` section.
- The appendix must start on a new page immediately after the references.
- Deliver only `report/` with the exact contents listed above.
- Do not create any other files.
- Do not invent, guess, or fill in missing information.
- If a detail cannot be verified from the available material, state explicitly that it could not be verified, or omit it if it is not necessary.
- Do not mention “this notebook”, “the code”, “the repository”, or similar.
- Avoid useless prose, repetition, and verbatim restatement of source text.

Writing requirements
- Be concise, but not artificially terse.
- Write enough to make the report pleasant to read and easy to follow.
- Prefer structure, whitespace, short paragraphs, and figures/tables over long prose.
- Keep the main body visually light: short sentences, compact paragraphs, sparse commentary.
- Do not include generic background, textbook explanations, or long implementation descriptions unless they are experimentally necessary and verifiable.
- Keep the appendix compact, formal, and strictly supportive of the main body.

Required structure

Main body:
- title
- abstract
- `\section{Introduction}`
- `\section{Setting}`
- `\section{Results}`
- `\section{Analysis}`

References:
- bibliography on a new page after `Analysis`

Appendix:
- `\appendix`
- `\section{Notation}`
- `\section{Metrics}`
- `\section{Algorithm}`

Content requirements

`Abstract`
- 3 to 6 sentences.
- State only the experimental setting, what is compared, and the main observed outcome.
- No hype, no novelty claims, no broad framing.

`Introduction`
- Short but readable.
- Provide enough context to understand the experiment and why the comparison matters.
- State the object of study, the comparison being made, and the scope of the reported evidence.
- Do not add literature review beyond a few strictly necessary citations.

`Setting`
Report only the verified experimental setup, as concretely as possible. Include, when available:
- target distribution, dataset, or simulation setup;
- compared methods or baselines;
- architecture / backbone;
- training hyperparameters and optimization details;
- evaluation protocol;
- reported metrics.

Keep this section short and factual. Prefer dense reporting over explanation, but allow enough wording for clarity.

`Results`
This section should carry most of the document.
- Take advantage of as much of the useful figure material as possible within the 10-page main-body limit.
- Reuse available outputs whenever possible.
- If needed, recreate figures/tables faithfully from the available material.
- If important results are present but not already visualized well, produce new figures or compact tables to display them clearly.
- Prefer informative figures and compact tables over text, as long as the 10-page main-body limit is respected.
- Use the strongest evidence first: final comparison plots, ablations, quantitative summaries, representative qualitative outputs, and compact tables.
- Exclude only figures that are redundant, low-information, purely diagnostic, or not needed to support the main findings.
- Include one compact “result overview” table near the beginning of the section whenever the available material supports it. This table should summarize the main quantitative comparison and serve as the anchor for the rest of the section.
- Every figure/table must have a short caption and a label.
- Each figure caption should end with one short factual takeaway sentence directly supported by the displayed evidence.
- Arrange content so the results are easy to scan visually.
- Add brief, useful commentary where needed to connect the figures and tables. The text should help interpretation, not duplicate the captions.

`Analysis`
- Keep this section short.
- Provide only pertinent remarks, compact interpretation, and useful intuition directly supported by the results.
- Do not paraphrase captions or restate the displayed evidence mechanically.
- Do not add generic claims, shallow commentary, or unsupported interpretation.
- Focus on what matters: strongest comparison, main trade-off, main failure mode, main ablation lesson, or other experimentally grounded insight.

`Notation`
- Provide a compact notation table covering only symbols actually used in the document.
- Prefer a two-column table: symbol and meaning.
- Do not include unused or generic notation.
- If notation cannot be verified, omit it rather than guessing.

`Metrics`
- Give formal mathematical definitions of every reported evaluation metric, but only for metrics actually used in the document.
- Keep definitions concise.
- Include variable meanings only when needed for clarity.
- If a metric is reported but its exact definition cannot be verified from the available material, state that the precise definition could not be verified.

`Algorithm`
- Give a compact formal definition of the model or process evaluated, limited to what is necessary for the reported experiments.
- For diffusion or flow-based models, include the forward formula(s) used in the reported setup, in mathematical form.
- If multiple forward parameterizations are compared, define each one briefly.
- Prefer equations over prose.
- Do not include derivations, training pseudocode, or background exposition unless experimentally necessary and verifiable.
- If the exact forward formulation cannot be verified, state that explicitly instead of reconstructing it from prior knowledge.

Figure and table policy
- `report/figures/` must contain only assets actually referenced in `report/main.tex`.
- Use as many non-redundant, informative figures as can fit cleanly within the 10-page main-body limit.
- Prefer multi-panel figures or compact layouts when this helps include more useful results without clutter.
- When existing figures are weak, incomplete, badly scaled, or poorly suited to the paper format, produce improved figures from the available data instead of forcing the original ones.
- Do not keep duplicate or near-duplicate visuals.
- Do not include raw screenshots unless they are the only faithful way to preserve a result.
- All figures must be properly scaled to fit the NeurIPS layout.
- No figure may overflow the text width, page height, or margins.
- Do not use oversized figures by default.
- Prefer single-column figures unless a two-column figure is clearly necessary.
- Prefer `[h]` placement for figures and tables whenever they fit cleanly at the current location.
- Use as many `[h]` placements as possible without breaking layout, creating large whitespace, or causing float problems.
- Fall back to `[t]`, `[tbp]`, or other standard float placements only when necessary for clean compilation and readable layout.
- Adjust widths, heights, aspect ratios, and subplot layouts so each figure is readable but compact.
- Prefer `width=\linewidth` or narrower for single-column content.
- Use full-width figures only when clearly justified by readability.
- If a figure becomes unreadable at the chosen scale, simplify it, crop unused whitespace aggressively, split it, or replace it with a more compact representation rather than letting it dominate the page.
- Tables must also be sized to fit cleanly within the page layout.

Bibliography policy
- Produce `report/bibliography.bib` and cite it from `report/main.tex`.
- Include only references actually cited in the document.
- The bibliography must integrate all references cited anywhere in the document, including the appendix.
- Keep the bibliography small.
- Add references only when they serve a clear purpose: essential context in the introduction, metric definitions, or algorithm/model definitions.
- Do not fabricate bibliographic metadata.
- If a citation is needed but full bibliographic details cannot be verified, either omit the citation or use a clearly minimal verified entry.
- Do not add a long related-work section.
- Place the bibliography on a new page after the `Analysis` section.

Style policy
- Concise but readable.
- No decorative phrasing.
- No long verbatim extraction from source text.
- No narrative filler.
- No unsupported interpretation.
- Let the figures and tables do most of the work, but allow enough prose to guide the reader through them.
- In the appendix, favor equations and compact tables over prose.

Final quality pass
Before finishing:
- Make one final pass over the document layout and reduce unnecessary whitespace.
- Remove any unused figure files.
- Ensure captions are short and informative.
- Ensure the main body stays within 4 pages.
- Ensure the references start on a new page after `Analysis`.
- Ensure the appendix starts on a new page after the references.
- Ensure all cited references are present in `bibliography.bib`.
- Ensure `main.tex` compiles cleanly with the copied `neurips_2026.sty`.

Final output contract
Return exactly one directory:
- `report/`

Ensure that:
- `report/main.tex` compiles with the copied `report/neurips_2026.sty`;
- `report/main.tex` references only files present in `report/figures/` and `report/bibliography.bib`;
- the references start on a new page after `Analysis`;
- the appendix starts on a new page after the references.
EOF
)

PROMPT="${PROMPT_TEMPLATE//__NOTEBOOK_BASENAME__/$NOTEBOOK_REF}"
PROMPT="${PROMPT//__STYLE_CONTENT__/$STYLE_CONTENT}"

codex exec \
  --cd "$(pwd)" \
  --sandbox workspace-write \
  "$PROMPT"
