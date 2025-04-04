%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------
% Tau
% LaTeX Template
% Version 2.4.4 (28/02/2025)
%
% Author: 
% Guillermo Jimenez (memo.notess1@gmail.com)
% 
% License:
% Creative Commons CC BY 4.0
% --------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tau-class/tau}[2025/02/28 tau class v2.4.4]

%----------------------------------------------------------
%	CLASS OPTIONS
%----------------------------------------------------------

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

%----------------------------------------------------------
%	REQUIRED PACKAGES
%----------------------------------------------------------

\RequirePackage{xcolor} % Color
\RequirePackage{graphicx} % Images
\RequirePackage{amsmath} % Math things
\RequirePackage{caption} % Caption editing
\RequirePackage{booktabs} % Better tables
\RequirePackage{enumitem} % Customize lists
\RequirePackage{tikz} % Draw shapes

\RequirePackage{fontawesome5} % Font icons
\RequirePackage{hyperref} % URLs
\RequirePackage{lettrine} % Drop caps
\RequirePackage{mdframed} % Frames
\RequirePackage{geometry} % Margins and page
\RequirePackage{datetime} % Display the date
\RequirePackage{datetime2} % Display the date
\RequirePackage{fancyhdr} % Headers
\RequirePackage{titlesec} % Title format
\RequirePackage{titling} % Title format
\RequirePackage{etoolbox} % Required for some font stuff
\RequirePackage{calc} % Calculations
\RequirePackage{lipsum} % For blind text
\RequirePackage{lineno} % Line numbers
\RequirePackage{listings} % Code display
\RequirePackage{stix2} % Font
\RequirePackage{tabularray} % Tables
\RequirePackage{subcaption} % Subfigures
\RequirePackage{float} % Set position of objects

%----------------------------------------------------------
%	BIBLATEX (uncomment to use)
%----------------------------------------------------------

\RequirePackage[
	backend=biber,
	style=ieee,
	sorting=ynt
]{biblatex}
\addbibresource{tau.bib}

%----------------------------------------------------------
%	GEOMETRY PACKAGE (MARGINS)
%----------------------------------------------------------

% Use this to change the margins later if needed
\geometry{hmargin=2.5cm,vmargin=2.5cm} % See geometry documentation for more options e.g. hmargin

%----------------------------------------------------------
% LISTINGS STYLE
%----------------------------------------------------------

\lstdefinestyle{taulst}{
  backgroundcolor=\color{white},   
  commentstyle=\color{gray},
  keywordstyle=\color{blue},
  numberstyle=\tiny\color{gray},
  stringstyle=\color{purple},
  basicstyle=\ttfamily\footnotesize,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=t,
  keepspaces=true,
  numbers=left,
  numbersep=5pt,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=2
}

\lstset{style=taulst,frame=single,lineskip=0.45em} 

%----------------------------------------------------------
% MATH PACKAGES
%----------------------------------------------------------

\RequirePackage{natbib}
\RequirePackage{amsfonts} % Math fonts
\RequirePackage{amsthm} % Math theorems
\RequirePackage{amssymb} % Math symbols
\RequirePackage{mathtools} % Math tools

%----------------------------------------------------------
% FONTS
%----------------------------------------------------------

\RequirePackage{fontspec} % Font selection
\RequirePackage{unicode-math} % Math unicode
\RequirePackage[english]{babel} % Language

%----------------------------------------------------------
% TITLE PREFERENCES
%----------------------------------------------------------

\newcommand{\titlepos}{\raggedright}
\newcommand{\journalname}[1]{\def\journalnametext{#1}}
\newcommand{\taustart}[1]{\lettrine{#1}{}}
\def\@maketitle{%
        \vspace*{1em}
        {\begin{center}\titlepos\LARGE\@title\end{center}}%
        \vspace{5pt}
        {\hrule height 2pt}
        \vspace{4pt}
        {\hrule height 0.8pt}
        \vspace{8pt}
        {\begin{center} \sc{\journalnametext}\end{center}}
        \vspace{8pt}
        {\begin{center}\@author\end{center}}
        \vspace{24pt}
     }

%----------------------------------------------------------
% TAUABSTRACT
%----------------------------------------------------------

% Define a keyword command
\newtoks\keywords
\newcommand\keywordname{\textbf{Keywords}}

% Define a new command to hold the professor name
\newcommand{\professor}[1]{\def\professorname{#1}}
\def\@professor{}

% Define the affiliation for the authors
\newcommand\affil[2][]{\if\relax\detokenize{#1}\relax\def\affil@arg{#2}\else\def\affil@arg{#1: #2}\fi
  \affil@i}
\def\affil@i{\@ifnextchar[\affil@ii{\affil@ii[]}}
\def\affil@ii[#1]{\stepcounter{affil}%
  \expandafter\def\csname affil@\theaffil\endcsname{\affil@arg}%
  \expandafter\def\csname affil@label@\theaffil\endcsname{#1}%
  \the\affils}
  
\newcounter{affil}
\newtoks\affils
\newcommand\printAffiliations{%
  \begin{center}
    \count@=0
    \loop\ifnum\count@<\c@affil
      \advance\count@ by 1 
      \csname affil@\the\count@\endcsname\par
    \repeat
    \professorname
  \end{center}%
}

% Adjust spacing for abstract
\renewenvironment{abstract}{%
    \if@twocolumn
      \section*{\abstractname}%
    \else %
      \begin{center}%
        {\bfseries\large \abstractname\vspace{-.5em}\vspace{\z@}}%
      \end{center}%
      \quotation
    \fi}
    {\if@twocolumn\else\endquotation\fi}

% New abstact command
\newcommand{\tauabstract}{%
    \begin{abstract} \@abstract\par\vspace{3ex}%
    \if\relax\detokenize\expandafter{\the\keywords}\relax\else
    {\noindent\keywordname: \the\keywords.}%
    \fi
    \end{abstract}%
    \vspace{-4ex}
    \printAffiliations
    \vspace{10ex}
}

\let\@abstract\relax
\AtBeginDocument{\let\@abstract\abstract \let\abstract\relax}

%----------------------------------------------------------
% EQUATION SKIP
%----------------------------------------------------------

\newlength{\eqskip}\setlength{\eqskip}{8pt}
	\expandafter\def\expandafter\normalsize\expandafter{%
		\normalsize%
		\setlength\abovedisplayskip{\eqskip}%
		\setlength\belowdisplayskip{\eqskip}%
		\setlength\abovedisplayshortskip{\eqskip-\baselineskip}%
		\setlength\belowdisplayshortskip{\eqskip}%
	}

%----------------------------------------------------------
% FOOTER INFORMATION
%----------------------------------------------------------

% Define commands for institution and sub footers
\newcommand{\institution}[1]{\def\institutionname{#1}}
\newcommand{\footinfo}[1]{\def\footinfoname{#1}}
\newcommand{\theday}[1]{\def\thedayname{#1}}
\newcommand{\leadauthor}[1]{\def\leadauthorname{#1}}
\newcommand{\course}[1]{\def\coursename{#1}}

% Define the footer style
\newcommand{\footrule}{\vspace{-1em}\hrule height 2pt \vspace{0.1em} \hrule height 0.5pt \vspace{0.5em}}

\newcommand{\authlabel}[1]{\textsuperscript{#1}}

\newcommand{\author}[2][]{\if\relax\detokenize{#1}\relax\def\author@arg{#2}\else\def\author@arg{#2\authlabel{#1}}\fi
  \author@i}
\def\author@i{\@ifnextchar[\author@ii{\author@ii[]}}
\def\author@ii[#1]{\stepcounter{auth}%
  \expandafter\def\csname auth@\theauth\endcsname{\author@arg}%
  \expandafter\def\csname auth@label@\theauth\endcsname{#1}%
  \the\auths}

\newcounter{auth}
\newtoks\auths
\renewcommand\@author{%
  \begin{center}
    \count@=0
    \loop\ifnum\count@<\c@auth
      \advance\count@ by 1 
      \csname auth@\the\count@\endcsname%
      \ifnum\count@<\c@auth
        \the\authsep
      \fi
    \repeat
  \end{center}%
}
\newtoks\authsep
\authsep={, }

% Set up fancy headers and footers
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[L]{\institutionname}
\fancyfoot[C]{\thepage}
\fancyfoot[R]{\coursename}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% First page style with extra content
\fancypagestyle{firststyle}{
  \fancyhf{}
  \lfoot{\institutionname}
  \cfoot{\thepage}
  \rfoot{\coursename}
  \lhead{\leadauthorname}
  \chead{\footinfoname}
  \rhead{\thedayname}
  \renewcommand{\headrulewidth}{0.0pt}
  \renewcommand{\footrulewidth}{0.0pt}
}

%----------------------------------------------------------
% TABLES
%----------------------------------------------------------

\newcommand{\tabletext}[1]{%
	\begin{center}
		\textit{#1}
	\end{center}
}

%----------------------------------------------------------
% ENVIRONMENTS PACKAGE
%----------------------------------------------------------

\RequirePackage{tau-class/tauenvs}

%----------------------------------------------------------
% BABEL PACKAGE
%----------------------------------------------------------

\RequirePackage{tau-class/taubabel}
\endinput