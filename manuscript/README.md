# ICASSP manuscript

Upload the contents of this directory to Overleaf and choose `main.tex` as the
main document, or compile locally:

```bash
pdflatex main.tex
pdflatex main.tex
```

The supplied `ParaSeedBench_ICASSP2027.pdf` has four technical pages followed by
a fifth references page. `main.tex` contains a page guard that raises a LaTeX
error if the references do not start on page five.

The table fragment `human_primary_results_table.tex` is the human-primary table
used by the manuscript. Numerical source files and validation commands live in
`../paper_artifacts/`.
