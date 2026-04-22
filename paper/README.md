# NeurIPS-workshop write-up

This folder contains the source for the project paper
*Fine-Tuning Language Models with Repeated Prompts* (main body targets
under 6 pages; references and appendix are additional).

## Layout

```
paper/
├── paper.tex              # main paper source (6-page body + appendix)
├── refs.bib               # bibliography
├── project_440_550.sty    # course-provided NeurIPS style
├── make_plots.py          # regenerates all figures from ../results_*.json
├── figures/               # pdf + png figures used by paper.tex
└── README.md              # this file
```

## Rebuilding the figures

```bash
python make_plots.py
```

This reads `../results_qwen2.5_1.5b.json`, `../results_mistral_7b.json`,
and `../results_qwen2.5_7b.json`, and writes one PDF and one PNG per
figure into `figures/`.

## Building the paper

Any recent TeXLive / MacTeX install with `biblatex` + `biber` works:

```bash
latexmk -pdf -bibtex paper.tex
# or, manually:
pdflatex paper.tex && biber paper && pdflatex paper.tex && pdflatex paper.tex
```

The style file is taken verbatim from `../instructions/project_440_550.sty`.
By default the paper is built with the `preprint` option (no line numbers,
no submission footer); remove `[preprint]` in `paper.tex` to get the
submission layout with line numbers.
