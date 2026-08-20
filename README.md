# oir-analysis-3

Re-analysis of **GSE150703** — mouse retina in the oxygen-induced retinopathy (OIR) model —
reading out **Txn1** expression across retinal cell types.

The whole analysis is one Quarto document, `oir_analysis.qmd`. It downloads the data, builds
the object, and then clusters, annotates and reads out two preparations, each written out in
full. There is no pipeline, no `bin/`, and no scripts.

**→ [`report.md`](report.md) is the rendered analysis, every figure inline.** Read it here in
the browser; nothing needs to be installed or downloaded to see the results.

## The data

One published DGE matrix, genes × cells, **already log1p-normalized** — the series publishes no
raw counts. Barcodes encode the design as `COND_TIMEPOINT_SORT_LAB_REP`, e.g.
`NORM_P14_Cd73ft_Joyal_1`, which is split into `obs` columns so the design travels with the data.

The `sort` × `lab` combinations are three preparations, and they are **never integrated**:

| Preparation | Cells | Analyzed |
|---|---|---|
| `WR_Joyal` | 15,143 | yes |
| `Cd73ft_Joyal` | 10,329 | yes |
| `WR_Mccarrol` | 5,799 | no |

Each is clustered on its own cells, so cluster numbers mean nothing across preparations and any
comparison between them happens at the level of cell types, not cluster IDs.

Cell types are called against **MRCA**, the mouse retina cell atlas (330,930 cells, all of it
disease-free, [doi:10.1016/j.isci.2024.109916](https://doi.org/10.1016/j.isci.2024.109916)),
downloaded from CELLxGENE alongside the DGE matrix.

## The document, in order

Three chunks are shared, and then each preparation gets the same run of chunks under its own
`#` heading, suffixed `-wr-joyal` or `-cd73ft-joyal`.

| Shared chunk | What it does |
|---|---|
| `setup` | imports and rcParams, nothing else |
| `download` | fetches the DGE matrix and the MRCA reference over HTTPS |
| `create-anndata` | reads the matrix, parses the barcodes, writes the full object |
| `reference-centroids` | mean profile per MRCA `majorclass`, written once and re-read |

| Per-preparation chunk | What it does |
|---|---|
| `cluster-*` | subsets, computes QC metrics, filters genes, embeds, sweeps Leiden 0.5–1.5 |
| `palettes-*` | design palettes, cluster colors, and the UMAP draw order |
| `umap-sweep-*` | the whole sweep as a UMAP grid |
| `choose-resolution-*` | the chosen resolution, and its clusters renumbered by size |
| `umap-clusters-*` | the chosen partition on the UMAP |
| `qc-violin-*` | QC metrics overall |
| `qc-violin-by-cluster-*` | QC metrics per cluster |
| `correlate-reference-*` | Spearman correlation, clusters × MRCA classes |
| `heatmap-reference-*` | that matrix raw, scaled per class, and scaled per cluster |
| `score-markers-*` | scores every cell against the retina marker panel |
| `score-violin-by-cluster-*` | scores per cluster |
| `umap-scores-*` | scores on the UMAP |
| `dotplot-markers-*` | marker genes by cluster |
| `heatmap-scores-*` | mean scores raw, scaled per cell type, and scaled per cluster |
| `annotate-clusters-*` | the call per cluster, with correlation, margin, and the marker call |
| `refine-cell-types-*` | splits any cluster the panel says holds two classes |
| `refine-scatter-*` | the two scores against each other, one panel per candidate split |
| `umap-cell-types-*` | the calls on the UMAP |
| `umap-txn1-*` | Txn1 on the UMAP |
| `txn1-violin-celltype-*` | Txn1 by cell type |
| `txn1-violin-condition-*` | Txn1 by cell type and condition |
| `txn1-violin-timepoint-*` | Txn1 by cell type and timepoint |
| `umap-txn1-stratified-*` | Txn1 on the UMAP, condition × timepoint |
| `txn1-violin-stratified-*` | Txn1 by cell type, NORM vs OIR within each timepoint |
| `txn1-violin-stratified-by-condition-*` | the same, P14 vs P17 within each condition |

New steps go at the end of a preparation's section, and then into the other one.

## Conventions

**Each preparation is written out in full rather than looped over.** The document is meant to
be read top to bottom, and a loop over slide functions meant following one figure by jumping
between the loop, a function and two helpers. The cost is real and worth stating: the analysis
appears twice, a change has to be made in both halves, and the marker panel and QC metric list
are each defined twice. Only the download, the AnnData build and the reference centroids are
shared.

**Steps write their output and skip if it exists.** The download, `create-anndata`,
`reference-centroids` and both `cluster-*` chunks each check for their file first and read it
back rather than recomputing. Deleting the file is how you force the work to happen again —
that is the whole invalidation mechanism, in place of Quarto `cache` or `freeze`.

The trap is worth stating: **once the output file exists, editing anything inside that chunk's
`else` branch does nothing.** Changing the sweep range, the gene filter, or a leiden argument
requires deleting the relevant `data/processed/*_clustered.h5ad` first.

Steps after the clustering are deliberately unguarded, so the resolution choice, the marker
panel, the override sheet, and every plot take effect on the next render with nothing to clear.

**`_v0` and `_v1`.** The sweep writes `leiden_res_<res>_v0`, leiden's own labels, kept for
audit. The chosen resolution gets a `_v1` column holding that same partition renumbered 1..k by
descending cell count, so cluster 1 is always the largest. `_v1` is what the plots read.

**Paths live in the chunk that uses them.** `setup` holds imports and rcParams only. A path
defined far from its use is misdirection.

## Decisions

**The sweep runs high (0.5–1.5), on purpose.** Annotation merges fine clusters into cell types,
so a cluster that is too fine still gets a coherent call, while one that is too coarse blends
two populations into a centroid nothing downstream can separate. Splitting is recoverable;
merging is not. oir-analysis-2 chose 0.4 for its objects, so cluster counts are not comparable
between the two projects. Both preparations here are read out at **1.0**.

**Leiden runs to convergence.** `n_iterations` is left at its `-1` default rather than the `2`
that scanpy's `FutureWarning` suggests: at 15k cells the difference is seconds, and it means the
extra clusters overclustering hands us are real structure rather than the algorithm stopping
early. `random_state=0` is passed although it is also the default, because it is the one
argument there that determines the output.

**Genes are filtered before HVG selection.** `filter_genes(min_cells=3)` — oir-analysis-2 left
this undone and flagged it. The subset otherwise carries every gene however few cells detect it,
and the `seurat` flavor z-scores dispersion within mean-expression bins, so near-empty genes
raise the bar in the low-mean bins they crowd.

**QC metrics are computed with `expr_type="log1p"`, and are not standard QC.** `X` is the
authors' normalized data, so `pct_log1p_mt` is *not* the mitochondrial fraction of UMIs — log
compresses high expressers, and a cell that is 40% mitochondrial by counts reads far lower here.
Use them to rank cells against each other, not against the usual thresholds. The authors
filtered before publishing; this describes their result rather than doing quality control on it.

### The reference

**The whole atlas is aggregated, not its P14/P17 cells.** That subset looks like the right match
for this design, but it holds no astrocytes at all, 50 RGCs, and bipolar cells almost only at
P17, so it cannot call several of the clusters here.

**Only three elements of the reference are read.** `read_h5ad` would also pull in `.raw`, a
second copy of an already 9.8 GB matrix, and take the load past this machine's memory. `X`,
`obs` and `var` are read directly and the sparse indices narrowed to `int32`.

**The correlation runs over the query's variable genes, not everything shared.** The
housekeeping bulk is near-identical in every class and only lifts all twelve correlations
together, flattening the differences the call is made on.

**Spearman**, so a class is matched on the order it puts the genes in rather than on absolute
levels — the two datasets were normalized by different pipelines.

**Nine symbols answer to more than one Ensembl id** in the atlas, and are dropped rather than
merged: there is no honest way to pick which id the query's column meant.

### The calls

**The call is the best-correlating reference class, then a human's override.** Each
`annotate-clusters-*` chunk carries a `cluster_call_overrides` sheet that shadows the argmax;
a cluster left out falls back to it. The sheet is keyed on that preparation's own size-ranked
cluster numbers and **does not transfer between preparations**.

`WR_Joyal` has an exhaustive sheet — every one of its 21 clusters written out after reading the
heatmap and the dotplot, so the argmax is fully shadowed and the sheet is a record rather than a
list of exceptions. As written it ratifies the reference everywhere, including the two clusters
the marker panel reads differently. `Cd73ft_Joyal`'s sheet is empty: that batch has not been
read by a human yet, so every call in it is still the argmax.

**The marker panel is an independent read, not the call.** Each score is a marker mean minus a
control set drawn from the same expression bins, so scores compare within a cell but mean
nothing on their own scale. The scores stay in the document — violins, UMAP panels, a dotplot
and a heatmap — so that where they and the reference disagree, the disagreement is visible
before an override is written.

The panel is inlined in the scoring chunk rather than read from a YAML file: it keeps this a
single document, avoids a `pyyaml` dependency, and puts the markers where whoever edits them is
already reading.

**A cluster can be split when the two vocabularies disagree.** The reference calls a whole
cluster one class, so a population that never forms its own cluster cannot be named — the
vascular cells in `WR_Joyal` are one cluster of 106 holding both endothelium and pericytes.
Where the panel names a cluster differently from the reference, that pair is a candidate, and
each cell goes to whichever of the two scores is higher provided it clears zero.

Which candidates are actually split is gated, because a disagreement says where to look, not
what to do. The test is the L in the scatter, written as a number: in a mixed cluster the two
scores trade off and their rank correlation across the cluster runs negative, while in one
population with a depth gradient both rise together and it runs positive. `max_tradeoff = -0.25`
sits between the vascular cluster's −0.75 and the next candidate, ten times weaker — the scores
have to genuinely trade off, not merely fail to correlate. `min_arm_cells` and
`min_arm_fraction` then throw out splits that shave off a handful of cells. A rejected candidate
keeps its whole-cluster call, and its row in the printed table is the record of the panel having
been overruled.

`WR_Joyal` screens clusters 10, 16 and 20 at −0.072, −0.747 and 0.214, so only 16 splits.
`Cd73ft_Joyal` screens 12 and 18 at −0.140 and −0.222 and splits neither — the `Cd73ft` sort
resolves endothelium and pericytes into clusters of their own, so there is no mixed vascular
cluster to recover there. The gate is WR_Joyal's, kept rather than retuned per batch.

## Setup

```bash
conda env create -f environment.yml
conda activate oir-analysis-3
python -m ipykernel install --user --name oir-analysis-3
```

The kernel registration is not optional. The document pins `jupyter: oir-analysis-3`, and Quarto
can only honour that if it is run by a python that can see the kernel — hence `QUARTO_PYTHON`
below. Without both, it renders against whatever kernelspec happens to be first and comes out
full of tracebacks.

Quarto itself is expected to be on `PATH` and is not installed by the env.

## Rendering

The document declares three formats — an HTML report, a PowerPoint deck, and the GitHub
markdown — so pick one:

```bash
export QUARTO_PYTHON=$CONDA_PREFIX/bin/python

quarto render oir_analysis.qmd --to html    # the working report
quarto render oir_analysis.qmd --to pptx    # the deck
quarto render oir_analysis.qmd --to gfm     # report.md, the copy GitHub renders
```

**Prefer `--to` over the bare `quarto render`**, which builds all three and executes the
notebook once per format — the whole analysis, three times.

The first render downloads the DGE matrix and the reference, spends a few minutes parsing and
aggregating them, and clusters both preparations; later renders read the saved objects instead.

The three formats differ only in what they show, not in what they run — **every step appears in
all of them**. HTML and gfm print their code; `echo: false` is scoped to the `pptx` block so
slides carry output alone. Text-only chunks are on slides too, which means the widest of them,
the 21 × 12 correlation table, is present but not readable from the back of a room.

**The gfm output is the one that is committed.** HTML and pptx are gitignored, being derived
and large, but `report.md` is what makes the analysis readable without building anything, so it
and its figures are tracked. A markdown file cannot inline its images the way `embed-resources`
lets the HTML, so the figures live beside it in `oir_analysis_files/figure-commonmark/` — 50
PNGs, about 9 MB. `.gitignore` reopens `*_files/` for that one subdirectory and no other.

The cost is worth knowing before you re-render: **every gfm render rewrites all 50 PNGs**, so
they show up as modified whether or not the analysis changed. Commit the figures when the
figures actually changed, and check out the rest — `git checkout -- oir_analysis_files/` after
a no-op re-render keeps the noise out of the history.

Slides come from `##` headings — pandoc has nothing else to split on, which is why every chunk
that produces output has one and `setup`, which produces none, does not. `slide-level: 2` is
required rather than stylistic: it is what keeps the two `#` preparation headings as section
breaks instead of letting pandoc infer a level that silently demotes every `##` to a bullet.

There is no reference template, so the deck uses pandoc's default 16:9 at 10 × 5.625in. Figure
sizes are set per chunk against that, and for scanpy figures they are **per panel** — scanpy
multiplies the figure size by `ncols`, so a `fig-width` meant as a whole-figure width silently
asks for something several times the slide.

A deck is a zip, and it is worth looking inside one rather than trusting an exit code:

```bash
mkdir -p /tmp/deck && unzip -o -q oir_analysis.pptx -d /tmp/deck
echo "slides: $(ls /tmp/deck/ppt/slides/*.xml | wc -l)  images: $(ls /tmp/deck/ppt/media | wc -l)"
```

The deck currently comes to 68 slides.

## Layout

```
├── oir_analysis.qmd    the analysis
├── report.md           the gfm render, committed so GitHub can show it
├── oir_analysis_files/
│   └── figure-commonmark/   report.md's figures, committed with it
├── environment.yml
├── data/raw/           downloaded from GEO and CELLxGENE, never edited (gitignored)
└── data/processed/     objects the document writes and re-reads       (gitignored)
```

`data/` is entirely reproducible from the document — deleting it costs a download and a rebuild,
nothing more.

## Not here yet

- `Cd73ft_Joyal` has no human-reviewed calls: its override sheet is empty and every call is the
  reference argmax. Its two clusters the panel disputes, 12 (`Astrocyte` vs `BC`) and 18
  (`Astrocyte` vs `MG`), are the place to start reading.
- `WR_Mccarrol` is not analyzed at all.
- Nothing is quantified. Txn1 is shown across cell type, condition and timepoint, but there is
  no test, no effect size, and no differential expression anywhere in the document.
- No integration, so the two preparations are compared only by eye, cell type by cell type.
