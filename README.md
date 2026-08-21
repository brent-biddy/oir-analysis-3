# oir-analysis-3

This repo contains analysis of Txn1 expression in the mouse retina.
Below are links to the reports produced by this repo:

- [**Txn1 Expression in the Mouse Retina**](reports/txn1-expression-retina.md) — clustering, cell
  type annotation, and visualization of Txn1 expression.

## Analysis Overview

We analyze a single-cell RNA seq data set from
[Binet *et al.* 2020](https://doi.org/10.1126/science.aay5356), deposited at
[GSE150703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150703). This consists of data
collected from two labs, Joyal and McCarroll, two tissue preps, Whole Retina and Cd73ft, two
timepoints p14 and p17, and two conditions, normoxia and oxygen-induced retinopathy.

We annotate this data using a mouse retina single cell atlas from
[Li *et al.* 2024](https://doi.org/10.1016/j.isci.2024.109916), which we pull from
[CELLxGENE](https://cellxgene.cziscience.com/). The reference consists of 330,930 cells across
twelve major cell types which we will try to use to annotate our data with: amacrine cells (AC),
astrocytes, bipolar cells (BC), cones, endothelial cells, horizontal cells (HC), Müller glia
(MG), microglia, pericytes, retinal ganglion cells (RGC), retinal pigment epithelium (RPE), and
rods. After annotation we inspect the expression of Txn1 across the different cell types in the
different conditions.

## Running The Analysis

### 1. Install conda

We use conda as the package manager for this analysis, so let's start by making sure it's
installed:

```bash
conda --version
```

If conda is not installed please see
[these instructions](https://docs.anaconda.com/miniconda/install/) for installing conda.

### 2. Create the conda environment

Now that conda is installed, let's use it to install the software the analysis needs.
`environment.yml` lists all of the software needed and we will use it to install these packages
with conda.

```bash
conda env create -f environment.yml
conda activate oir-analysis-3
```

### 3. Register the Jupyter kernel

Now that the conda environment has been created, we need to register the jupyter kernel in the
environment so it can be discovered by quarto.

```bash
python -m ipykernel install --user --name oir-analysis-3
```

### 4. Render the Analysis

With the conda environment installed and activated we can now render the analysis. This creates
two files in `reports/`, a github markdown document and a powerpoint deck. The powerpoint deck is
not included in the repo.

```bash
quarto render txn1-expression-retina.qmd --output-dir reports
```

The first render downloads the data, then parses and clusters it, which takes a few minutes. The
downloads and the objects the analysis builds along the way are saved to `data/`, and later
renders skip those steps if the file is already saved. To re-run one of these steps we delete the
file it saved:

```bash
rm data/processed/GSE150703_adata_WR_Joyal_clustered.h5ad    # to change the gene filter, leiden args
```

## Layout

```
├── txn1-expression-retina.qmd    the analysis
├── fold-code.lua                 folds the code chunks in the gfm render
├── environment.yml
├── reports/                      everything the document renders to
│   ├── txn1-expression-retina.md         the gfm render, committed so GitHub shows it
│   └── txn1-expression-retina_files/     its figures
└── data/                         downloads and saved objects (gitignored, reproducible)
```
