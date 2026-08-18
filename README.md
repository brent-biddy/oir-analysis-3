# oir-analysis-3

Re-analysis of **GSE150703** — mouse retina in the oxygen-induced retinopathy (OIR) model —
reading out **Txn1** expression across retinal cell types.

The whole analysis is one Quarto document, `oir_analysis.qmd`. It downloads the data, builds
the object, clusters one preparation, and plots what is needed to judge that clustering. There
is no pipeline, no `bin/`, and no scripts.

## The data

One published DGE matrix, genes × cells, **already log1p-normalized** — the series publishes no
raw counts. Barcodes encode the design as `COND_TIMEPOINT_SORT_LAB_REP`, e.g.
`NORM_P14_Cd73ft_Joyal_1`, which is split into `obs` columns so the design travels with the data.

The `sort` × `lab` combinations are three preparations, and they are **never integrated**:

| Preparation | Cells |
|---|---|
| `WR_Joyal` | 15,143 |
| `Cd73ft_Joyal` | 10,329 |
| `WR_Mccarrol` | 5,799 |

Each is clustered on its own cells, so cluster numbers mean nothing across preparations and any
comparison between them happens at the level of cell types, not cluster IDs. The document
currently analyzes `WR_Joyal`. The others get their own chunks when their turn comes — named
for the preparation, not parameterized over it.

## The document, in order

| Chunk | What it does |
|---|---|
| `setup` | imports, nothing else |
| `download` | fetches the DGE matrix from GEO over HTTPS |
| `create-anndata` | reads the matrix, parses the barcodes, writes the full object |
| `cluster-wr-joyal` | subsets to WR_Joyal, computes QC metrics, filters genes, embeds, sweeps Leiden 0.5–1.5 |
| `umap-sweep-wr-joyal` | the whole sweep as a UMAP grid |
| `choose-resolution-wr-joyal` | the chosen resolution, and its clusters renumbered by size |
| `umap-clusters-wr-joyal` | the chosen partition on the UMAP |
| `qc-violin-wr-joyal` | QC metrics overall |
| `qc-violin-by-cluster-wr-joyal` | QC metrics per cluster |
| `score-markers-wr-joyal` | scores every cell against the retina marker panel |
| `score-violin-by-cluster-wr-joyal` | scores per cluster |
| `umap-scores-wr-joyal` | scores on the UMAP |
| `dotplot-markers-wr-joyal` | marker genes by cluster |

New steps go at the end.

## Conventions

**Steps write their output and skip if it exists.** The download and `create-anndata` and
`cluster-wr-joyal` chunks each check for their file first and read it back rather than
recomputing. Deleting the file is how you force the work to happen again — that is the whole
invalidation mechanism, in place of Quarto `cache` or `freeze`.

The trap is worth stating: **once the output file exists, editing anything inside that chunk's
`else` branch does nothing.** Changing the sweep range, the gene filter, or a leiden argument
requires deleting `data/processed/GSE150703_adata_WR_Joyal_clustered.h5ad` first.

Steps after the clustering are deliberately unguarded, so the resolution choice, the marker
panel, and every plot take effect on the next render with nothing to clear.

**`_v0` and `_v1`.** The sweep writes `leiden_res_<res>_v0`, leiden's own labels, kept for
audit. The chosen resolution gets a `_v1` column holding that same partition renumbered 1..k by
descending cell count, so cluster 1 is always the largest. `_v1` is what the plots read.

**Paths live in the chunk that uses them.** `setup` holds imports only. A path defined far from
its use is misdirection.

## Decisions

**The sweep runs high (0.5–1.5), on purpose.** Annotation merges fine clusters into cell types,
so a cluster that is too fine still gets a coherent call, while one that is too coarse blends
two populations into a centroid nothing downstream can separate. Splitting is recoverable;
merging is not. oir-analysis-2 chose 0.4 for its objects, so cluster counts are not comparable
between the two projects.

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

**Nothing is called a cell type.** `score_genes` writes eleven `score_*` columns and stops
there. Each score is a marker mean minus a control set drawn from the same expression bins, so
scores compare within a cell but mean nothing on their own scale, and an argmax over eleven
four-gene sets would label every cell whether or not any set fits it. The scores are shown as
evidence — violins by cluster, UMAP panels, and a dotplot — and the call is left to a human.

The marker panel is inlined in the scoring chunk rather than read from a YAML file: it keeps
this a single document, avoids a `pyyaml` dependency, and puts the markers where whoever edits
them is already reading.

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

The document declares two formats — an HTML report and a PowerPoint deck — so pick one:

```bash
export QUARTO_PYTHON=$CONDA_PREFIX/bin/python

quarto render oir_analysis.qmd --to html    # the working report
quarto render oir_analysis.qmd --to pptx    # the deck
```

**Prefer `--to` over the bare `quarto render`**, which builds both formats and executes the
notebook once per format — the whole analysis, twice.

The first render downloads 236 MB, spends a few minutes parsing it, and clusters; later renders
read the saved objects instead. Both outputs are gitignored.

The two formats differ only in what they show, not in what they run. The HTML prints its code;
`echo: false` is scoped to the `pptx` block so slides carry figures alone.

Slides come from `##` headings — pandoc has nothing else to split on, which is why every chunk
that produces output has one and `setup`, which produces none, does not. `slide-level: 2` is
required rather than stylistic: without it pandoc infers the level from the headings present,
and adding a first `#` would silently demote every `##` to a bullet.

There is no reference template, so the deck uses pandoc's default 16:9 at 10 × 5.625in. Figure
sizes are set per chunk against that, and they are **per panel** — scanpy multiplies the
figure size by `ncols`, so a `fig-width` meant as a whole-figure width silently asks for
something several times the slide.

A deck is a zip, and it is worth looking inside one rather than trusting an exit code:

```bash
mkdir -p /tmp/deck && unzip -o -q oir_analysis.pptx -d /tmp/deck
echo "slides: $(ls /tmp/deck/ppt/slides/*.xml | wc -l)  images: $(ls /tmp/deck/ppt/media | wc -l)"
```

## Layout

```
├── oir_analysis.qmd    the analysis
├── environment.yml
├── data/raw/           downloaded from GEO, never edited      (gitignored)
└── data/processed/     objects the document writes and re-reads (gitignored)
```

`data/` is entirely reproducible from the document — deleting it costs a download and a rebuild,
nothing more.

## Not here yet

- Only `WR_Joyal` is analyzed; `Cd73ft_Joyal` and `WR_Mccarrol` are not.
- No cell type calls, and no correlation against an external reference atlas — oir-analysis-2
  used the Lukowski human retina atlas for that, with the caveat that it is neural retina only
  and so has no vascular endothelial or pericyte population to match against.
- No Txn1 readout yet, which is the point of the exercise.
