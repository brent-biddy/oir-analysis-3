# oir-analysis-3

Re-analysis of **GSE150703** — mouse retina in the oxygen-induced retinopathy (OIR) model —
reading out **Txn1** expression across retinal cell types.

## The data

[GSE150703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150703) (Binet *et al.*,
*Science* **369**, eaay5356, 2020) is normalized single-cell counts from mouse retina. Each
barcode carries the experimental design in its name — condition (normoxia or OIR), timepoint
(P14 or P17), dissociation prep, lab, and replicate:

```
OIR_P17_WR_Joyal_r1_AGCTATCAATTT
│   │   │  │     │  └── droplet barcode
│   │   │  │     └───── replicate
│   │   │  └─────────── lab
│   │   └────────────── prep
│   └────────────────── timepoint
└────────────────────── condition
```

Prep and lab together make the batch, and the two batches analyzed here are `WR_Joyal`, whole
retina, and `Cd73ft_Joyal`, rod-depleted.

## The analysis

Setup downloads the counts and builds them into an AnnData object, splitting each barcode into
those design columns. From there each batch is subset out and analyzed on its own, both the same
way: QC metrics, gene filtering, variable genes, PCA, neighbors, UMAP, and Leiden at a fixed
resolution.

Clusters are named against a reference — the mouse retina cell atlas (Li *et al.*, *iScience*
**27**, 109916, 2024), 330,930 healthy retina cells across twelve major cell types, pulled from
CELLxGENE and summed into one centroid per cell type. Every query cell is Spearman-correlated
against all twelve centroids over the shared genes, and the correlations are standardized within
each cell so the numbers compare between cells rather than track sequencing depth. Each cluster
takes its best-correlating class; a refinement pass then re-labels cells inside a cluster that
consistently prefer a different class, so populations that never formed their own cluster still
get named.

With cell types in hand, the rest is the Txn1 readout: Txn1 on the UMAP and as violins, broken
out by cell type, by condition, by timepoint, and across the full condition-by-timepoint design.

Everything lives in one Quarto document, `oir_analysis.qmd`. There is no pipeline and no scripts.

**→ [`report.md`](report.md) is the rendered analysis, every figure inline.** Nothing needs to be
installed to read it.

## Running it

### 1. Install conda

We use conda as the package manager for this analysis, so let's start by making sure it's
installed:

```bash
conda --version
```

If that doesn't answer, install Miniconda first — the instructions are at
[docs.anaconda.com/miniconda/install](https://docs.anaconda.com/miniconda/install/).

### 2. Create the environment

Now that conda is installed, let's use it to install the software the analysis needs.
`environment.yml` lists all of it — python, scanpy and the rest of the scverse stack, and Quarto
alongside them, so the Quarto that renders the document is the one in the environment rather than
whatever happens to be on `PATH`.

```bash
conda env create -f environment.yml
conda activate oir-analysis-3
```

### 3. Register the Jupyter kernel

Quarto runs the document's code through a Jupyter kernel, and the document asks for one by name:
`jupyter: oir-analysis-3`. So let's register the environment we just made under that name.

```bash
python -m ipykernel install --user --name oir-analysis-3
```

Without this, Quarto falls back to whatever kernel comes first and fills the output with
tracebacks, so it's worth doing before the first render rather than after.

### 4. Render

Now we can render. The document builds three formats, and we'll ask for one at a time — a bare
`quarto render` builds all three and re-executes the whole analysis once per format.

```bash
quarto render oir_analysis.qmd --to html    # the working report
quarto render oir_analysis.qmd --to pptx    # the deck
quarto render oir_analysis.qmd --to gfm     # report.md, the copy GitHub renders
```

The first render downloads ~3.7 GB — 226 MB of counts from GEO and a 3.5 GB atlas from
CELLxGENE — then parses and clusters for a few minutes. Between the downloads and the objects
the document saves alongside them, `data/` ends up around 4.3 GB. Later renders read those saved
objects back and skip the work.

### 5. Re-running a step

The download, `create-anndata`, `reference-centroids` and both `cluster-*` chunks skip themselves
if their output file already exists. That means editing anything inside those chunks does nothing
until we delete what they wrote:

```bash
rm data/processed/GSE150703_adata_WR_Joyal_clustered.h5ad    # to change the gene filter, leiden args
```

Everything after the clustering is unguarded and takes effect on the next render.

Re-rendering gfm is safe to do freely. Every seed is pinned and matplotlib writes no timestamps,
so a render that changes nothing produces byte-identical figures and git reports no diff. Only
the figures that actually changed show up.

## Layout

```
├── oir_analysis.qmd         the analysis
├── report.md                the gfm render, committed so GitHub shows it
├── oir_analysis_files/      report.md's figures
├── environment.yml
└── data/                    downloads and saved objects (gitignored, reproducible)
```
