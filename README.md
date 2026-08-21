# oir-analysis-3

This repo contains the analysis of Txn1 expression in the mouse retina.
Here are links to the reports in this repo:

- [**The analysis**](reports/txn1-expression-retina.md) — every step and every figure, rendered
  for GitHub.

## The data

We analyze a single-cell RNA seq data set from
[Binet *et al.* 2020](https://doi.org/10.1126/science.aay5356), deposited at
[GSE150703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150703). This consists of data
collected from two labs, Joyal and McCarroll, two tissue preps, Whole Retina and Cd73ft, two
timepoints p14 and p17, and two conditions, normoxia and oxygen-induced retinopathy.

We annotate this data using a mouse retina single cell atlas from
[Li *et al.* 2024](https://doi.org/10.1016/j.isci.2024.109916), which we pull from
[CELLxGENE](https://cellxgene.cziscience.com/). The reference consists of 330,930 cells across
twelve major cell types which we will try to use to annotate our data with. After annotation we
inspect the expression of Txn1 across the different cell types in the different conditions.

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

With the conda environment installed and activated we can now render the analysis. One command
builds both formats the document declares: a github formatted markdown document, which is the
copy GitHub shows, and a powerpoint deck.

```bash
quarto render txn1-expression-retina.qmd --output-dir reports
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

## Layout

```
├── txn1-expression-retina.qmd    the analysis
├── environment.yml
├── reports/                      everything the document renders to
│   ├── txn1-expression-retina.md         the gfm render, committed so GitHub shows it
│   └── txn1-expression-retina_files/     its figures
└── data/                         downloads and saved objects (gitignored, reproducible)
```
