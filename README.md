# oir-analysis-3

Re-analysis of **GSE150703** — mouse retina in the oxygen-induced retinopathy (OIR) model —
reading out **Txn1** expression across retinal cell types.

The whole analysis is one Quarto document, `oir_analysis.qmd`. No pipeline, no scripts.

**→ [`report.md`](report.md) is the rendered analysis, every figure inline.** Nothing needs to
be installed to read it.

## Setup

```bash
conda env create -f environment.yml
conda activate oir-analysis-3
python -m ipykernel install --user --name oir-analysis-3
```

The kernel registration is not optional — the document pins `jupyter: oir-analysis-3`, and
without it Quarto renders against whatever kernel comes first and fills the output with
tracebacks. Quarto itself is expected on `PATH` and is not installed by the env.

## Rendering

```bash
export QUARTO_PYTHON=$CONDA_PREFIX/bin/python

quarto render oir_analysis.qmd --to html    # the working report
quarto render oir_analysis.qmd --to pptx    # the deck
quarto render oir_analysis.qmd --to gfm     # report.md, the copy GitHub renders
```

Pass `--to`. The bare `quarto render` builds all three formats and re-executes the whole
analysis once per format.

The first render downloads ~236 MB from GEO and CELLxGENE, then parses and clusters for a few
minutes. Later renders read the saved objects in `data/processed/`.

## Gotchas

**Deleting the output file is how you re-run a step.** The download, `create-anndata`,
`reference-centroids` and both `cluster-*` chunks skip themselves if their file already exists,
so editing anything inside those chunks does nothing until you delete what they wrote:

```bash
rm data/processed/GSE150703_adata_WR_Joyal_clustered.h5ad    # to change the sweep, gene filter, leiden args
```

Everything after the clustering is unguarded and takes effect on the next render.

**A gfm render rewrites all 50 figures**, changed or not. After a no-op re-render:

```bash
git checkout -- oir_analysis_files/
```

## Layout

```
├── oir_analysis.qmd         the analysis
├── report.md                the gfm render, committed so GitHub shows it
├── oir_analysis_files/      report.md's figures
├── environment.yml
└── data/                    downloads and saved objects (gitignored, reproducible)
```

## Status

`WR_Joyal` and `Cd73ft_Joyal` are analyzed; `WR_Mccarrol` is not. `Cd73ft_Joyal`'s cell types
are unreviewed reference argmax — its override sheet is empty. Nothing about Txn1 is quantified.
