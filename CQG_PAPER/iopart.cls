%
% iopart.cls (shim)
%
% CQG historically used the IOP "iopart" class. This workspace currently
% ships the newer iopjournal.cls template but not iopart.cls. To allow
% compilation of legacy iopart-based sources without external downloads,
% we provide a lightweight compatibility shim that delegates formatting to
% iopjournal.cls and aliases the small set of macros used by the paper.
%
% If you have the official iopartcls package available, you can delete this
% shim and use the real iopart.cls instead.
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iopart}[2026/02/14 iopart shim delegating to iopjournal.cls]

% Pass all options through to iopjournal.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{iopjournal}}
\ProcessOptions\relax

\LoadClass{iopjournal}

\makeatletter

% iopart supports \title[short]{long}; iopjournal's \title takes one arg.
\let\iop@shim@orig@title\title
\renewcommand{\title}{\@ifnextchar[{\iop@shim@title@opt}{\iop@shim@orig@title}}
\def\iop@shim@title@opt[#1]#2{\iop@shim@orig@title{#2}}

% iopjournal prints the title block at the point \title/\author/\affil are used.
% A legacy \maketitle call should therefore be a no-op.
\renewcommand{\maketitle}{}

% Aliases for iopart macros used in the source.
\@ifundefined{address}{\newcommand{\address}[1]{\affil{#1}}}{}
\@ifundefined{ead}{\newcommand{\ead}[1]{\email{#1}}}{}
\@ifundefined{submitto}{\newcommand{\submitto}[1]{}}{}

% Provide an iopart-like "indented" list environment if missing.
\@ifundefined{indented}{%
  \newenvironment{indented}{%
    \begin{list}{}{%
      \setlength{\leftmargin}{1.5em}%
      \setlength{\rightmargin}{1.5em}%
      \setlength{\topsep}{0pt}%
      \setlength{\partopsep}{0pt}%
    }%
  }{\end{list}}%
}{}

\makeatother

