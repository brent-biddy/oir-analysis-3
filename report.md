# OIR retina scRNA-seq
Brent Biddy
2026-08-20

- [Setup](#setup)
- [Whole Retina](#whole-retina)
- [Rod-depleted Retina](#rod-depleted-retina)

# Setup

## Load Libraries

First, let’s load the libraries we need. We’ll also set a few default
plotting parameters here, so the figures stay consistent throughout the
document.

``` python
from pathlib import Path                    # used to construct file paths
from urllib.request import urlretrieve      # used to download query and reference data

import anndata as ad                        # used to load data
import matplotlib.colors as mcolors         # used to convert palette colours to hex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import csr_matrix         # used to store the counts sparsely

plt.rcParams.update({
    "figure.dpi": 200,        # resolution of the figures as drawn
    "savefig.dpi": 200,       # resolution of the figures as written out
    "font.size": 12,          # font size of the on-data cluster numbers, and of dotplot text
    "axes.labelsize": 12,     # font size of scanpy's UMAP axis labels
    "axes.titlesize": 13,     # font size of scanpy's panel titles
    "ytick.labelsize": 10.5,  # font size of the heatmap colorbar ticks
    "legend.fontsize": 11,    # font size of scanpy's UMAP legends
})
```

## Download Data

Now let’s download the two datasets we need: the OIR and normoxia retina
counts from GEO, and the mouse retina cell atlas we’ll use as an
annotation reference. Both are saved to `data/raw/`, and later renders
re-use them rather than downloading again.

The query counts are already normalized, so we’ll use them as they are
rather than normalizing them ourselves.

- **Query** —
  [GSE150703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150703).
  Binet *et al.*, Neutrophil extracellular traps target senescent
  vasculature for tissue remodeling in retinopathy. *Science* **369**,
  eaay5356 (2020).
  [doi:10.1126/science.aay5356](https://doi.org/10.1126/science.aay5356)
- **Reference** — the mouse retina cell atlas (MRCA), via CELLxGENE. Li
  *et al.*, Comprehensive single-cell atlas of the mouse retina.
  *iScience* **27**, 109916 (2024).
  [doi:10.1016/j.isci.2024.109916](https://doi.org/10.1016/j.isci.2024.109916)

``` python
source_urls = {
    "query": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE150nnn/GSE150703/suppl/"
        "GSE150703_retina_NORM_OIR_P14_P17_C57_WR_CD73FT_noamg_normalizedUMI_Count_DGEmatrix.txt.gz"
    ),
    "reference": (
        "https://datasets.cellxgene.cziscience.com/"
        "a420c2bf-feeb-48db-a6c7-71f492f23131.h5ad"
    ),
}

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

source_paths = {}
for source_name, source_url in source_urls.items():
    source_path = raw_dir / Path(source_url).name
    if not source_path.exists():               # downloaded once, then re-used on later renders
        urlretrieve(source_url, source_path)
    source_paths[source_name] = source_path
    print(f"{source_name}: {source_path.name}  ({source_path.stat().st_size / 1e6:.0f} MB)")

query_path = source_paths["query"]
reference_path = source_paths["reference"]
```

    query: GSE150703_retina_NORM_OIR_P14_P17_C57_WR_CD73FT_noamg_normalizedUMI_Count_DGEmatrix.txt.gz  (236 MB)
    reference: a420c2bf-feeb-48db-a6c7-71f492f23131.h5ad  (3690 MB)

## Build Query AnnData

Next, let’s build an AnnData object from the query counts. The matrix
comes as genes by cells, so we’ll transpose it and store it sparsely.

Each barcode also encodes the experimental design, so we’ll split it
into columns:

    OIR_P17_WR_Joyal_r1_AGCTATCAATTT
    │   │   │  │     │  └── droplet barcode
    │   │   │  │     └───── replicate
    │   │   │  └─────────── lab
    │   │   └────────────── prep
    │   └────────────────── timepoint
    └────────────────────── condition

We’ll also combine prep and lab into a batch column, since that pairing
is what we’ll cluster on.

``` python
adata_path = Path("data/processed/GSE150703_adata.h5ad")
adata_path.parent.mkdir(parents=True, exist_ok=True)

if adata_path.exists():                        # built once, then read back on later renders
    adata = sc.read_h5ad(adata_path)
else:
    counts = pd.read_csv(query_path, sep="\t", index_col=0).T   # genes x cells, so transposed

    adata = ad.AnnData(X=csr_matrix(counts.values))
    adata.obs_names = counts.index.astype(str)
    adata.var_names = counts.columns.astype(str)

    fields = pd.Series(adata.obs_names).str.split("_", expand=True)  # split barcode into parts
    adata.obs["condition"] = pd.Categorical(fields[0].values)
    adata.obs["timepoint"] = pd.Categorical(fields[1].values)
    adata.obs["prep"] = pd.Categorical(fields[2].values)
    adata.obs["lab"] = pd.Categorical(fields[3].values)
    adata.obs["replicate"] = pd.Categorical(fields[4].values)
    adata.obs["batch"] = pd.Categorical(fields[2].str.cat(fields[3], sep="_").values)  # prep x lab

    adata.write_h5ad(adata_path)

# outside the guard, so a colour can be changed and re-rendered without rebuilding anything
obs_palettes = {
    "condition": {"NORM": "#2AABB8", "OIR": "#E87D2A"},
    "timepoint": {"P14": "#2855A0", "P17": "#D63650"},
    "prep":      {"Cd73ft": "#C9A227", "WR": "#5DB85D"},
    "lab":       {"Joyal": "#7B4FB5", "Mccarrol": "#A0522D"},
    "batch":     {"Cd73ft_Joyal": "#C9A227", "WR_Joyal": "#5DB85D", "WR_Mccarrol": "#A0522D"},
}

for obs_key, palette in obs_palettes.items():
    adata.uns[f"{obs_key}_colors"] = [palette[c] for c in adata.obs[obs_key].cat.categories]

adata
```

    AnnData object with n_obs × n_vars = 31271 × 21408
        obs: 'condition', 'timepoint', 'prep', 'lab', 'replicate', 'batch'
        uns: 'condition_colors', 'timepoint_colors', 'prep_colors', 'lab_colors', 'batch_colors'
        layers: None (.X)

The object is saved to `data/processed/` and re-used on later renders.

## Build Reference Centroids

Now let’s build the reference we’ll annotate against. The mouse retina
cell atlas holds 330,930 cells across twelve major cell types, all of it
healthy retina.

To make the centroids we’ll sum the UMI counts for each gene across the
cells of each cell type, and do the same for CP10K normalized counts.
We’ll keep both as layers, along with the number of cells in each type,
so we can calculate the mean later.

``` python
centroid_path = Path("data/processed/MRCA_majorclass_centroids.h5ad")

if centroid_path.exists():                     # built once, then read back on later renders
    reference_centroids = sc.read_h5ad(centroid_path)
else:
    backed = sc.read_h5ad(reference_path, backed="r")   # nothing read into memory yet
    reference = ad.AnnData(
        X=backed.raw.X.to_memory(),            # the atlas keeps its raw counts in .raw
        obs=backed.obs[["majorclass"]],
        var=backed.raw.var[["feature_name"]],
    )

    reference.var["ensembl_id"] = reference.var_names   # kept before the index becomes symbols
    reference.var_names = reference.var.pop("feature_name").str.upper()  # the query uses symbols
    reference.var_names_make_unique(join="_v")  # nine symbols answer to more than one Ensembl id

    counts_sum = sc.get.aggregate(reference, by="majorclass", func="sum")
    sc.pp.normalize_total(reference, target_sum=1e4)   # counts -> CP10K, in place
    cp10k_sum = sc.get.aggregate(reference, by="majorclass", func="sum")
    del reference

    reference_centroids = ad.AnnData(
        X=cp10k_sum.layers["sum"],
        obs=pd.DataFrame(index=cp10k_sum.obs_names),
        var=cp10k_sum.var,
        layers={"counts": counts_sum.layers["sum"]},
    )
    reference_centroids.obs["n_cells"] = cp10k_sum.obs["n_obs_aggregated"].to_numpy()

    reference_centroids.write_h5ad(centroid_path)

reference_centroids
```

    AnnData object with n_obs × n_vars = 12 × 31671
        obs: 'n_cells'
        var: 'ensembl_id'
        layers: 'counts', None (.X)

# Whole Retina

Now let’s work through the whole retina prep, the `WR_Joyal` batch. It
holds 15,143 cells across both conditions and both timepoints, though
not evenly:

``` python
wr_joyal_obs = adata.obs[adata.obs["batch"] == "WR_Joyal"]
design = pd.crosstab(wr_joyal_obs["condition"], wr_joyal_obs["timepoint"])
print(design.rename_axis(index=None, columns=None).to_markdown())   # a real table, not a repr
```

|      |  P14 |  P17 |
|:-----|-----:|-----:|
| NORM | 2512 |  932 |
| OIR  | 5821 | 5878 |

## Cluster Cells

First we’ll subset the query data down to the `WR_Joyal` batch, and
analyze the batch’s cells in isolation.

Then we’ll compute QC metrics, filter out rarely detected genes, keep
the most variable genes, and embed. Rather than commit to one resolution
up front, we’ll cluster across a sweep of resolutions and choose from it
in the next step.

``` python
wr_joyal_resolution = 0.40   # chosen in oir-analysis and oir-analysis-2
leiden_key = f"leiden_res_{wr_joyal_resolution:.2f}_v0"
ranked_key = f"leiden_res_{wr_joyal_resolution:.2f}_v1"

wr_joyal_clustered_path = Path("data/processed/GSE150703_adata_WR_Joyal_clustered.h5ad")

if wr_joyal_clustered_path.exists():           # clustered once, then read back on later renders
    wr_joyal = sc.read_h5ad(wr_joyal_clustered_path)
else:
    wr_joyal = adata[adata.obs["batch"] == "WR_Joyal"].copy()   # this batch only

    for column in ["condition", "timepoint", "prep", "lab", "replicate", "batch"]:
        wr_joyal.obs[column] = wr_joyal.obs[column].cat.remove_unused_categories()  # drop empty levels

    wr_joyal.var["mt"] = wr_joyal.var_names.str.startswith("MT-")
    wr_joyal.var["ribo"] = wr_joyal.var_names.str.startswith(("RPS", "RPL"))
    wr_joyal.var["hb"] = wr_joyal.var_names.str.startswith(("HBA-", "HBB-"))
    sc.pp.calculate_qc_metrics(
        wr_joyal,
        qc_vars=["mt", "ribo", "hb"],
        expr_type="log1p",                     # X is normalized, so these are not UMI fractions
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    sc.pp.filter_genes(wr_joyal, min_cells=3)  # rarely detected genes distort the variable set
    sc.pp.highly_variable_genes(wr_joyal, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(wr_joyal, svd_solver="arpack")
    sc.pp.neighbors(wr_joyal, n_neighbors=10, n_pcs=40)
    sc.tl.umap(wr_joyal)

    sc.tl.leiden(
        wr_joyal,
        resolution=wr_joyal_resolution,
        key_added=leiden_key,
        flavor="igraph",
        random_state=0,                        # the one argument here that decides the output
    )

    # renumbered by size, so cluster 1 is always the largest
    cluster_sizes = wr_joyal.obs[leiden_key].value_counts()
    ranked_labels = [str(rank) for rank in range(1, len(cluster_sizes) + 1)]
    wr_joyal.obs[ranked_key] = (
        wr_joyal.obs[leiden_key]
        .map(dict(zip(cluster_sizes.index, ranked_labels)))
        .astype("category")
        .cat.reorder_categories(ranked_labels)
    )

    for obs_key, palette in obs_palettes.items():
        categories = wr_joyal.obs[obs_key].cat.categories
        wr_joyal.uns[f"{obs_key}_colors"] = [palette[category] for category in categories]

    wr_joyal.uns[f"{ranked_key}_colors"] = [               # tab20, cycled once past 20 clusters
        mcolors.to_hex(plt.cm.tab20.colors[(int(label) - 1) % 20]) for label in ranked_labels
    ]

    wr_joyal.write_h5ad(wr_joyal_clustered_path)

wr_joyal
```

    AnnData object with n_obs × n_vars = 15143 × 19084
        obs: 'condition', 'timepoint', 'prep', 'lab', 'replicate', 'batch', 'n_genes_by_log1p', 'total_log1p', 'total_log1p_mt', 'pct_log1p_mt', 'total_log1p_ribo', 'pct_log1p_ribo', 'total_log1p_hb', 'pct_log1p_hb', 'leiden_res_0.40_v0', 'leiden_res_0.40_v1'
        var: 'mt', 'ribo', 'hb', 'n_cells_by_log1p', 'mean_log1p', 'pct_dropout_by_log1p', 'total_log1p', 'n_cells', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
        uns: 'batch_colors', 'condition_colors', 'hvg', 'lab_colors', 'leiden_res_0.40_v0', 'leiden_res_0.40_v1_colors', 'neighbors', 'pca', 'prep_colors', 'timepoint_colors', 'umap'
        obsm: 'X_pca', 'X_umap'
        varm: 'PCs'
        obsp: 'connectivities', 'distances'
        layers: None (.X)

The clustered object is saved to `data/processed/`, so later renders
read it back and skip the clustering entirely. To re-run any of it,
delete that file first.

## Clusters on the UMAP

``` python
# scanpy builds the figure here, and takes no figsize argument, so rcParams is where its size
# has to be set
plt.rcParams["figure.figsize"] = (5.0, 4.3)

ax = sc.pl.umap(
    wr_joyal,
    color=ranked_key,
    legend_loc="on data",
    legend_fontoutline=2,
    frameon=True,
    show=False,
)
ax.set_aspect("equal", adjustable="datalim")
plt.show()
```

<img
src="oir_analysis_files/figure-commonmark/umap-clusters-wr-joyal-output-1.png"
id="umap-clusters-wr-joyal" />

## QC metrics

``` python
qc_metrics = {
    "n_genes_by_log1p": "Genes detected",
    "total_log1p": "Total expression",
    "pct_log1p_mt": "% mitochondrial",
    "pct_log1p_ribo": "% ribosomal",
    "pct_log1p_hb": "% hemoglobin",
}
violin_inner_kws = {"box_width": 2, "whis_width": 1, "marker": "o",
                    "markersize": 3.0, "markeredgecolor": "white"}

qc_df = sc.get.obs_df(wr_joyal, keys=[ranked_key] + list(qc_metrics))
batch_color = wr_joyal.uns["batch_colors"][0]

fig, axes = plt.subplots(1, len(qc_metrics), squeeze=False, figsize=(9, 3.0),
                         constrained_layout=True)
for ax, (qc_key, qc_label) in zip(axes.flat, qc_metrics.items()):
    sns.violinplot(data=qc_df, y=qc_key, color=batch_color, cut=0, density_norm="width",
                   inner="box", inner_kws=violin_inner_kws, linewidth=0.5, ax=ax)
    ax.set_ylabel(qc_label, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/qc-violin-wr-joyal-output-1.png"
id="qc-violin-wr-joyal" />

## QC metrics by cluster

``` python
# only scanpy reads the _colors entry out of uns, so the palette is handed to seaborn
# explicitly — otherwise a cluster changes color between the UMAP and these violins
cluster_palette = dict(zip(wr_joyal.obs[ranked_key].cat.categories, wr_joyal.uns[f"{ranked_key}_colors"]))

fig, axes = plt.subplots(len(qc_metrics), 1, squeeze=False, sharex=True, figsize=(9, 4.3),
                         constrained_layout=True)
for ax, (qc_key, qc_label) in zip(axes.flat, qc_metrics.items()):
    # dodge=False because hue repeats x: left on, seaborn gives each of the 21 clusters its
    # own sub-slot and every violin is drawn a twentieth of its slot wide, off its tick
    sns.violinplot(data=qc_df, x=ranked_key, y=qc_key, hue=ranked_key, palette=cluster_palette,
                   legend=False, dodge=False, cut=0, density_norm="width", inner="box",
                   inner_kws=violin_inner_kws, linewidth=0.5, saturation=1, ax=ax)

    # seaborn colours the bodies in an order that does not follow the x positions when hue
    # repeats x, so each is recoloured from the cluster it actually sits under. The order
    # comes off the categorical rather than the tick labels, which are empty on a shared axis.
    cluster_order = list(wr_joyal.obs[ranked_key].cat.categories)
    for body in ax.collections:
        extents = body.get_paths()[0].get_extents()
        body.set_facecolor(cluster_palette[cluster_order[round((extents.x0 + extents.x1) / 2)]])

    ax.set_ylabel(qc_label, fontsize=7)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)

axes[-1][0].set_xlabel("Cluster", fontsize=9)
axes[-1][0].tick_params(axis="x", labelsize=8)
plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/qc-violin-by-cluster-wr-joyal-output-1.png"
id="qc-violin-by-cluster-wr-joyal" />

## Correlation with the reference

``` python
# correlated over the query's variable genes rather than everything shared: the
# housekeeping bulk is near-identical in every class and only lifts all twelve correlations
# together, flattening the differences the call is made on
feature_genes = [
    gene for gene in wr_joyal.var_names[wr_joyal.var["highly_variable"]]
    if gene in reference_centroids.var_names
]

cluster_profiles = pd.DataFrame(
    [
        np.asarray(
            wr_joyal[wr_joyal.obs[ranked_key] == cluster, feature_genes].X.mean(axis=0)
        ).ravel()
        for cluster in wr_joyal.obs[ranked_key].cat.categories
    ],
    index=wr_joyal.obs[ranked_key].cat.categories,
    columns=feature_genes,
)
reference_profiles = pd.DataFrame(
    reference_centroids[:, feature_genes].X,
    index=reference_centroids.obs_names,
    columns=feature_genes,
)

# Spearman, so a class is matched on the order it puts the genes in rather than on absolute
# levels — the two datasets were normalized by different pipelines. Ranking the rows and
# taking Pearson is the same thing, and does all 21 x 12 pairs in one call.
ranked_clusters = cluster_profiles.rank(axis=1)
ranked_reference = reference_profiles.rank(axis=1)
reference_correlation = pd.DataFrame(
    np.corrcoef(ranked_clusters, ranked_reference)[:len(ranked_clusters), len(ranked_clusters):],
    index=cluster_profiles.index,
    columns=reference_profiles.index,
)

print(f"{len(feature_genes)} of {int(wr_joyal.var['highly_variable'].sum())} "
      f"variable genes found in the reference")
reference_correlation.round(3)
```

    1618 of 2000 variable genes found in the reference

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | AC | Astrocyte | BC | Cone | Endothelial | HC | MG | Microglia | Pericyte | RGC | RPE | Rod |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1 | 0.626 | 0.584 | 0.677 | 0.798 | 0.464 | 0.630 | 0.656 | 0.473 | 0.500 | 0.589 | 0.617 | 0.861 |
| 2 | 0.742 | 0.648 | 0.814 | 0.730 | 0.538 | 0.702 | 0.727 | 0.526 | 0.586 | 0.685 | 0.600 | 0.729 |
| 3 | 0.842 | 0.645 | 0.719 | 0.658 | 0.515 | 0.707 | 0.703 | 0.513 | 0.572 | 0.739 | 0.564 | 0.668 |
| 4 | 0.703 | 0.641 | 0.777 | 0.732 | 0.540 | 0.685 | 0.715 | 0.552 | 0.580 | 0.654 | 0.606 | 0.738 |
| 5 | 0.635 | 0.749 | 0.614 | 0.562 | 0.607 | 0.596 | 0.865 | 0.560 | 0.620 | 0.578 | 0.612 | 0.583 |
| 6 | 0.624 | 0.561 | 0.681 | 0.832 | 0.452 | 0.639 | 0.638 | 0.471 | 0.499 | 0.591 | 0.614 | 0.795 |
| 7 | 0.776 | 0.616 | 0.690 | 0.638 | 0.497 | 0.701 | 0.669 | 0.522 | 0.541 | 0.716 | 0.541 | 0.644 |
| 8 | 0.715 | 0.590 | 0.633 | 0.605 | 0.463 | 0.671 | 0.641 | 0.498 | 0.504 | 0.764 | 0.505 | 0.609 |
| 9 | 0.528 | 0.727 | 0.493 | 0.440 | 0.616 | 0.484 | 0.755 | 0.505 | 0.623 | 0.479 | 0.645 | 0.469 |
| 10 | 0.435 | 0.547 | 0.430 | 0.392 | 0.760 | 0.414 | 0.554 | 0.497 | 0.720 | 0.404 | 0.462 | 0.384 |
| 11 | 0.641 | 0.552 | 0.633 | 0.607 | 0.471 | 0.752 | 0.615 | 0.485 | 0.523 | 0.627 | 0.531 | 0.602 |
| 12 | 0.359 | 0.540 | 0.375 | 0.375 | 0.510 | 0.373 | 0.443 | 0.778 | 0.442 | 0.364 | 0.410 | 0.378 |

</div>

## Correlation with the reference by cluster

``` python
# the same matrix three ways, matching the marker heatmap. The raw correlations sit in a
# narrow band, so the two scaled panels are what can actually be read: down a column asks
# "which cluster is most this class", and across a row asks "which class best fits this
# cluster" — the annotation question, and the one the calls are taken from.
#
# only the row scaling is a call. A column's brightest cell is the best home for that class
# whether or not the class is present at all: RPE peaks on a rod cluster here, which says
# where it would land rather than that it was found.
class_range = reference_correlation.max(axis=0) - reference_correlation.min(axis=0)
scaled_by_class = reference_correlation.sub(
    reference_correlation.min(axis=0), axis=1
).div(class_range, axis=1)

correlation_range = reference_correlation.max(axis=1) - reference_correlation.min(axis=1)
scaled_correlation = reference_correlation.sub(
    reference_correlation.min(axis=1), axis=0
).div(correlation_range, axis=0)

correlation_panels = {
    "Spearman correlation": reference_correlation,
    "Scaled per class": scaled_by_class,
    "Scaled per cluster": scaled_correlation,
}

# walk the clusters in order and take each one's best class; a class already placed is
# skipped, and any class no cluster leads with is appended. The matches then read as a
# diagonal rather than being scattered across the panel.
class_order = []
for cluster in scaled_correlation.index:
    best = scaled_correlation.loc[cluster].idxmax()
    if best not in class_order:
        class_order.append(best)
class_order += [name for name in scaled_correlation.columns if name not in class_order]

fig, axes = plt.subplots(1, len(correlation_panels), squeeze=False, figsize=(9, 4.3),
                         constrained_layout=True)
for ax, (title, matrix) in zip(axes.flat, correlation_panels.items()):
    # xticklabels/yticklabels left at "auto" thins the labels to every other one once a
    # third panel is on the figure, which leaves half the classes unnamed
    sns.heatmap(matrix[class_order], cmap="viridis", linewidths=0.5, linecolor="white",
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "pad": 0.02}, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Reference class", fontsize=8)
    ax.set_ylabel("Cluster", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/heatmap-reference-wr-joyal-output-1.png"
id="heatmap-reference-wr-joyal" />

## Correlation with the reference by cell

Correlating each cell against the same centroids says which clusters
hold more than one cell type, without needing a marker panel to name the
second one.

``` python
# every shared gene here, where the cluster-level correlation above uses variable genes only.
# A centroid is dense, so its housekeeping bulk lifts all twelve classes together and flattens
# the differences; a single cell is 97% zero over those same variable genes and needs every
# gene it can get. Widening to all of them moves agreement with the cluster call from 90% to 95%.
shared_genes = [gene for gene in wr_joyal.var_names if gene in reference_centroids.var_names]

ranked_cells = pd.DataFrame(np.asarray(wr_joyal[:, shared_genes].X.todense())).rank(axis=1).to_numpy()
ranked_classes = pd.DataFrame(
    reference_centroids[:, shared_genes].X, index=reference_centroids.obs_names
).rank(axis=1).to_numpy()

# Spearman again, written out rather than looped so all twelve classes come out of one product
ranked_cells = ranked_cells - ranked_cells.mean(axis=1, keepdims=True)
ranked_classes = ranked_classes - ranked_classes.mean(axis=1, keepdims=True)
cell_correlation = (ranked_cells @ ranked_classes.T) / np.sqrt(
    (ranked_cells ** 2).sum(axis=1)[:, None] * (ranked_classes ** 2).sum(axis=1)[None, :]
)

wr_joyal.obs["cell_call"] = pd.Categorical(
    reference_centroids.obs_names[cell_correlation.argmax(axis=1)],
    categories=sorted(reference_centroids.obs_names),
)

cell_composition = pd.crosstab(wr_joyal.obs[ranked_key], wr_joyal.obs["cell_call"])
(cell_composition.loc[:, cell_composition.sum() > 0]
 .div(cell_composition.sum(axis=1), axis=0).mul(100).round().astype(int))
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

| cell_call | AC | Astrocyte | BC | Cone | Endothelial | HC | MG | Microglia | Pericyte | RGC | RPE | Rod |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| leiden_res_0.40_v1 |  |  |  |  |  |  |  |  |  |  |  |  |
| 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 97 |
| 2 | 1 | 0 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| 3 | 90 | 0 | 4 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 3 |
| 4 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 1 | 1 | 0 | 0 | 0 | 97 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 1 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| 9 | 0 | 47 | 0 | 0 | 0 | 0 | 39 | 0 | 0 | 0 | 14 | 0 |
| 10 | 0 | 0 | 0 | 0 | 39 | 0 | 0 | 0 | 60 | 0 | 1 | 0 |
| 11 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 | 1 |

</div>

## Marker scores

``` python
retina_markers = {
    "Rods": ["Rho", "Pde6b", "Nrl", "Nr2e3"],
    "Cones": ["Opn1mw", "Opn1sw", "Arr3", "Thrb"],
    "RGC": ["Rbpms", "Sncg", "Pou4f1", "Pou4f2"],
    "Amacrine": ["Tfap2a", "Tfap2b", "Chat", "Vsx1"],
    "Bipolar": ["Vsx2", "Isl1", "Prkca", "Cabp5"],
    "Horizontal": ["Onecut1", "Onecut2", "Lhx1", "Prox1"],
    "Muller_Glia": ["Rlbp1", "Glul", "Sox9", "Slc1a3"],
    "Microglia": ["Cx3cr1", "P2ry12", "Tmem119", "Aif1"],
    "Vascular_Endothelial": ["Cdh5", "Pecam1", "Tek"],
    "Pericytes": ["Pdgfrb", "Acta2", "Des"],
    "Astrocytes": ["Gfap", "Aldh1l1", "Aqp4"],
}

score_keys = []
markers_present = {}
for cell_type, genes in retina_markers.items():
    present = [gene.upper() for gene in genes if gene.upper() in wr_joyal.var_names]
    score_key = f"score_{cell_type}"
    sc.tl.score_genes(wr_joyal, gene_list=present, score_name=score_key)
    score_keys.append(score_key)
    markers_present[cell_type] = present
    print(f"{cell_type}: {len(present)}/{len(genes)} markers")
```

    Rods: 4/4 markers
    Cones: 4/4 markers
    RGC: 4/4 markers
    Amacrine: 4/4 markers
    Bipolar: 4/4 markers
    Horizontal: 4/4 markers
    Muller_Glia: 4/4 markers
    Microglia: 4/4 markers
    Vascular_Endothelial: 3/3 markers
    Pericytes: 3/3 markers
    Astrocytes: 3/3 markers

## Marker scores by cluster

``` python
score_df = sc.get.obs_df(wr_joyal, keys=[ranked_key] + score_keys)

for start in range(0, len(score_keys), 6):
    group = score_keys[start:start + 6]
    group_types = list(retina_markers)[start:start + 6]

    fig, axes = plt.subplots(len(group), 1, squeeze=False, sharex=True, figsize=(9, 4.3),
                             constrained_layout=True)
    for ax, score_key, cell_type in zip(axes.flat, group, group_types):
        sns.violinplot(data=score_df, x=ranked_key, y=score_key, hue=ranked_key,
                       palette=cluster_palette, legend=False, dodge=False, cut=0,
                       density_norm="width", inner="box", inner_kws=violin_inner_kws,
                       linewidth=0.5, saturation=1, ax=ax)

        cluster_order = list(wr_joyal.obs[ranked_key].cat.categories)
        for body in ax.collections:
            extents = body.get_paths()[0].get_extents()
            body.set_facecolor(cluster_palette[cluster_order[round((extents.x0 + extents.x1) / 2)]])

        ax.set_ylabel(f"{cell_type.replace('_', ' ')} score", fontsize=7)
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[-1][0].set_xlabel("Cluster", fontsize=9)
    axes[-1][0].tick_params(axis="x", labelsize=8)
    plt.show()
    plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/score-violin-by-cluster-wr-joyal-output-1.png"
id="score-violin-by-cluster-wr-joyal-1" />

<img
src="oir_analysis_files/figure-commonmark/score-violin-by-cluster-wr-joyal-output-2.png"
id="score-violin-by-cluster-wr-joyal-2" />

## Marker scores on the UMAP

``` python
for start in range(0, len(score_keys), 3):
    group = score_keys[start:start + 3]

    fig, axs = plt.subplots(2, 3, height_ratios=[1, 0.22], figsize=(8.6, 3.4),
                            constrained_layout=True)
    for ax, bar_ax, score_key in zip(axs[0], axs[1], group):
        sc.pl.umap(
            wr_joyal,
            color=score_key,
            ax=ax,
            frameon=True,
            colorbar_loc=None,
            show=False,
        )
        ax.set_aspect("equal", adjustable="datalim")

        # the strip reserves the space; the colorbar drawn inside it is excluded from the
        # layout so its tick labels cannot push the panel around
        bar_ax.axis("off")
        colorbar_ax = bar_ax.inset_axes([0.325, 0.82, 0.35, 0.18])
        colorbar_ax.set_in_layout(False)
        colorbar = fig.colorbar(ax.collections[0], cax=colorbar_ax, orientation="horizontal")

        # placed explicitly at the ends and middle: the automatic locator picks its own round
        # numbers per score, so the tick count and positions differ from panel to panel
        score_low, score_high = ax.collections[0].get_clim()
        colorbar_ticks = np.linspace(score_low, score_high, 3)
        colorbar.set_ticks(colorbar_ticks, labels=[f"{tick:.2f}" for tick in colorbar_ticks])
        colorbar.ax.tick_params(labelsize=7)

    for empty_ax in axs[0][len(group):]:
        fig.delaxes(empty_ax)
    for empty_bar_ax in axs[1][len(group):]:
        empty_bar_ax.axis("off")

    plt.show()
    plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/umap-scores-wr-joyal-output-1.png"
id="umap-scores-wr-joyal-1" />

<img
src="oir_analysis_files/figure-commonmark/umap-scores-wr-joyal-output-2.png"
id="umap-scores-wr-joyal-2" />

<img
src="oir_analysis_files/figure-commonmark/umap-scores-wr-joyal-output-3.png"
id="umap-scores-wr-joyal-3" />

<img
src="oir_analysis_files/figure-commonmark/umap-scores-wr-joyal-output-4.png"
id="umap-scores-wr-joyal-4" />

## Marker genes by cluster

``` python
# figsize is deliberately not passed: dotplot derives its own from the gene and cluster counts,
# and forcing it to slide dimensions packs 41 genes across 21 clusters until the dots overlap
sc.pl.dotplot(wr_joyal, markers_present, groupby=ranked_key, standard_scale="var")
```

<img
src="oir_analysis_files/figure-commonmark/dotplot-markers-wr-joyal-output-1.png"
id="dotplot-markers-wr-joyal" />

## Marker score heatmap

``` python
mean_scores = score_df.groupby(ranked_key, observed=True)[score_keys].mean()
mean_scores.columns = [cell_type.replace("_", " ") for cell_type in retina_markers]

# the same matrix three ways, because they answer different questions: the raw means show
# magnitude, scaling down a column asks "which cluster is most this cell type", and scaling
# across a row asks "which cell type best fits this cluster" — the annotation question
scaled_by_type = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min())
row_range = mean_scores.max(axis=1) - mean_scores.min(axis=1)
scaled_by_cluster = mean_scores.sub(mean_scores.min(axis=1), axis=0).div(row_range, axis=0)

score_normalizations = {
    "Mean score": mean_scores,
    "Scaled per cell type": scaled_by_type,
    "Scaled per cluster": scaled_by_cluster,
}

# walk the clusters in order and take each one's best cell type; a type already placed is
# skipped, and any type no cluster leads with is appended. The matches then read as a
# diagonal rather than being scattered across the panel.
column_order = []
for cluster in scaled_by_cluster.index:
    best = scaled_by_cluster.loc[cluster].idxmax()
    if best not in column_order:
        column_order.append(best)
column_order += [c for c in scaled_by_cluster.columns if c not in column_order]

fig, axes = plt.subplots(1, len(score_normalizations), squeeze=False, figsize=(9, 4.3),
                         constrained_layout=True)
for ax, (title, matrix) in zip(axes.flat, score_normalizations.items()):
    # as on the reference heatmap: left at "auto" the labels thin to every other one across
    # three panels, which leaves half the cell types and half the clusters unnamed
    sns.heatmap(matrix[column_order], cmap="viridis", linewidths=0.5, linecolor="white",
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "pad": 0.02}, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Cell type", fontsize=8)
    ax.set_ylabel("Cluster", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/heatmap-scores-wr-joyal-output-1.png"
id="heatmap-scores-wr-joyal" />

## Preliminary cell types

``` python
# the call is the best-correlating reference class. The marker scores stay in the document
# as an independent read on the same question — where the two disagree, the disagreement is
# the thing to look at before writing an override.
cluster_calls = reference_correlation.idxmax(axis=1)
marker_calls = mean_scores.idxmax(axis=1)

# no calls made by hand yet, so every cluster below is the reference argmax
cluster_call_overrides = {}
cluster_calls.update(pd.Series(cluster_call_overrides))

wr_joyal.obs["cell_type"] = wr_joyal.obs[ranked_key].map(cluster_calls).astype("category")

# keyed on every class in the reference rather than only the ones called, so a cell type
# keeps its colour whatever a given resolution happens to find
cell_type_palette = dict(zip(sorted(reference_centroids.obs_names), sc.pl.palettes.default_20))
wr_joyal.uns["cell_type_colors"] = [
    cell_type_palette[cell_type] for cell_type in wr_joyal.obs["cell_type"].cat.categories
]

# margin is the gap to the runner-up class: a cluster called on a hair's difference is one
# to read the heatmap for, however high its correlation
sorted_correlation = np.sort(reference_correlation.to_numpy(), axis=1)

pd.DataFrame({
    "cell_type": cluster_calls,
    "cells": wr_joyal.obs[ranked_key].value_counts(),
    "correlation": reference_correlation.max(axis=1).round(3),
    "margin": (sorted_correlation[:, -1] - sorted_correlation[:, -2]).round(3),
    "marker call": marker_calls,
})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | cell_type   | cells | correlation | margin | marker call |
|-----|-------------|-------|-------------|--------|-------------|
| 1   | Rod         | 5613  | 0.861       | 0.064  | Rods        |
| 2   | BC          | 2952  | 0.814       | 0.072  | Bipolar     |
| 3   | AC          | 2029  | 0.842       | 0.103  | Amacrine    |
| 4   | BC          | 1768  | 0.777       | 0.040  | Bipolar     |
| 5   | MG          | 1230  | 0.865       | 0.116  | Muller Glia |
| 6   | Cone        | 700   | 0.832       | 0.036  | Cones       |
| 7   | AC          | 299   | 0.776       | 0.060  | Amacrine    |
| 8   | RGC         | 140   | 0.764       | 0.049  | RGC         |
| 9   | MG          | 120   | 0.755       | 0.028  | Muller Glia |
| 10  | Endothelial | 106   | 0.760       | 0.040  | Pericytes   |
| 11  | HC          | 101   | 0.752       | 0.111  | Horizontal  |
| 12  | Microglia   | 85    | 0.778       | 0.238  | Microglia   |

</div>

## Refining a call with the per-cell correlations

``` python
# the reference calls a whole cluster one class, so a population that never forms its own
# cluster cannot be named from the cluster call alone. The per-cell correlations above say
# where that happened: a class a real share of a cluster's cells correlate best with is a
# population inside it, and those cells take that class instead of the cluster's.
#
# a share rather than a count, because a handful of cells favouring a class is what dropout
# does to every cluster; and a count as well, so a small cluster cannot qualify a class on
# three cells. RPE takes 14% of one cluster here and is excluded by the share, which is the
# right answer for a class with no business in a neural retina prep.
min_share = 0.25
min_cells = 10

refinements = []
for cluster in cell_composition.index:
    counts = cell_composition.loc[cluster]
    populations = counts[(counts >= min_cells) & (counts / counts.sum() >= min_share)]
    in_cluster = (wr_joyal.obs[ranked_key] == cluster).to_numpy()

    for population in populations.index:
        if population not in wr_joyal.obs["cell_type"].cat.categories:
            wr_joyal.obs["cell_type"] = wr_joyal.obs["cell_type"].cat.add_categories([population])
        # a cell keeps the cluster's call unless its own best class is one that qualified
        takes = in_cluster & (wr_joyal.obs["cell_call"] == population).to_numpy()
        wr_joyal.obs.loc[takes, "cell_type"] = population

    refinements.append({
        "cluster": cluster,
        "call": cluster_calls[cluster],
        "cells": int(in_cluster.sum()),
        "populations": ", ".join(f"{name} {n}" for name, n in populations.items()),
        "refined": bool(len(populations) > 1 or populations.index[0] != cluster_calls[cluster]),
    })

# a cluster with one qualifying population that is already its call is left exactly as it was
print(pd.DataFrame(refinements).to_string(index=False))

# a class added above lands at the end of the categories, which would put it last on every
# axis downstream rather than in with the rest
wr_joyal.obs["cell_type"] = wr_joyal.obs["cell_type"].cat.remove_unused_categories()
wr_joyal.obs["cell_type"] = wr_joyal.obs["cell_type"].cat.reorder_categories(
    sorted(wr_joyal.obs["cell_type"].cat.categories)
)
wr_joyal.uns["cell_type_colors"] = [
    cell_type_palette[cell_type] for cell_type in wr_joyal.obs["cell_type"].cat.categories
]
```

    cluster        call  cells                 populations  refined
          1         Rod   5613                    Rod 5440    False
          2          BC   2952                     BC 2837    False
          3          AC   2029                     AC 1819    False
          4          BC   1768                     BC 1759    False
          5          MG   1230                     MG 1194    False
          6        Cone    700                    Cone 692    False
          7          AC    299                      AC 299    False
          8         RGC    140                     RGC 140    False
          9          MG    120         Astrocyte 56, MG 47     True
         10 Endothelial    106 Endothelial 41, Pericyte 64     True
         11          HC    101                      HC 101    False
         12   Microglia     85                Microglia 84    False

## Cluster composition by cell

``` python
# every cluster's per-cell composition as one bar. A cluster that is one cell type is a single
# block; a mixed one is divided, and the dashed line is the share a class has to clear to be
# treated as a population rather than as dropout.
shares = cell_composition.div(cell_composition.sum(axis=1), axis=0)
shares = shares.loc[:, shares.max() > 0.01]

fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
left = np.zeros(len(shares))
for population in shares.columns:
    ax.barh(shares.index.astype(str), shares[population], left=left,
            color=cell_type_palette[population], label=population, height=0.8)
    left += shares[population].to_numpy()

ax.axvline(min_share, color="#404040", linewidth=1.0, linestyle="--", zorder=3)
ax.set_xlabel("Share of the cluster's cells", fontsize=9)
ax.set_ylabel("Cluster", fontsize=9)
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.tick_params(labelsize=8)
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/refine-scatter-wr-joyal-output-1.png"
id="refine-scatter-wr-joyal" />

## Preliminary cell types on the UMAP

``` python
plt.rcParams["figure.figsize"] = (6.5, 4.3)

ax = sc.pl.umap(
    wr_joyal,
    color="cell_type",
    frameon=True,
    show=False,
)
ax.set_aspect("equal", adjustable="datalim")
plt.show()
```

<img
src="oir_analysis_files/figure-commonmark/umap-cell-types-wr-joyal-output-1.png"
id="umap-cell-types-wr-joyal" />

## Txn1 on the UMAP

``` python
fig, axs = plt.subplots(2, 1, height_ratios=[1, 0.22], figsize=(4.3, 3.9),
                        constrained_layout=True)

sc.pl.umap(wr_joyal, color="TXN1", ax=axs[0], frameon=True, colorbar_loc=None, show=False)
axs[0].set_aspect("equal", adjustable="datalim")

axs[1].axis("off")
colorbar_ax = axs[1].inset_axes([0.325, 0.82, 0.35, 0.18])
colorbar_ax.set_in_layout(False)
colorbar = fig.colorbar(axs[0].collections[0], cax=colorbar_ax, orientation="horizontal")

txn1_low, txn1_high = axs[0].collections[0].get_clim()
colorbar_ticks = np.linspace(txn1_low, txn1_high, 3)
colorbar.set_ticks(colorbar_ticks, labels=[f"{tick:.2f}" for tick in colorbar_ticks])
colorbar.ax.tick_params(labelsize=7)

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/umap-txn1-wr-joyal-output-1.png"
id="umap-txn1-wr-joyal" />

## Txn1 by cell type

``` python
txn1_df = sc.get.obs_df(wr_joyal, keys=["TXN1", "cell_type", "condition", "timepoint"])

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
sns.violinplot(data=txn1_df, x="cell_type", y="TXN1", hue="cell_type",
               palette=cell_type_palette, legend=True, dodge=False, cut=0,
               density_norm="width", inner="box", inner_kws=violin_inner_kws,
               linewidth=0.5, saturation=1, ax=ax)

# with hue repeating x and dodge off, seaborn hands the bodies their colours in an order
# that does not follow the x positions — the legend ends up right and the violins wrong.
# Recolour each body from the cell type it actually sits under.
cell_type_order = list(wr_joyal.obs["cell_type"].cat.categories)
for body in ax.collections:
    extents = body.get_paths()[0].get_extents()
    body.set_facecolor(cell_type_palette[cell_type_order[round((extents.x0 + extents.x1) / 2)]])

ax.set_xlabel("")
ax.set_ylabel("Txn1 (log1p)", fontsize=9)
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
ax.set_axisbelow(True)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-celltype-wr-joyal-output-1.png"
id="txn1-violin-celltype-wr-joyal" />

## Txn1 by cell type and condition

``` python
condition_palette = dict(zip(wr_joyal.obs["condition"].cat.categories,
                             wr_joyal.uns["condition_colors"]))

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
sns.violinplot(data=txn1_df, x="cell_type", y="TXN1", hue="condition",
               palette=condition_palette, cut=0, density_norm="width", inner="box",
               inner_kws=violin_inner_kws, linewidth=0.5, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Txn1 (log1p)", fontsize=9)
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
ax.set_axisbelow(True)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-condition-wr-joyal-output-1.png"
id="txn1-violin-condition-wr-joyal" />

## Txn1 by cell type and timepoint

``` python
timepoint_palette = dict(zip(wr_joyal.obs["timepoint"].cat.categories,
                             wr_joyal.uns["timepoint_colors"]))

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
sns.violinplot(data=txn1_df, x="cell_type", y="TXN1", hue="timepoint",
               palette=timepoint_palette, cut=0, density_norm="width", inner="box",
               inner_kws=violin_inner_kws, linewidth=0.5, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Txn1 (log1p)", fontsize=9)
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
ax.set_axisbelow(True)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-timepoint-wr-joyal-output-1.png"
id="txn1-violin-timepoint-wr-joyal" />

## Txn1 across the design

``` python
# laid out like the violin slides below: timepoint down the rows, condition across the
# columns, so the same comparison sits in the same direction in every Txn1 figure
condition_order = list(wr_joyal.obs["condition"].cat.categories)
timepoints = list(wr_joyal.obs["timepoint"].cat.categories)

umap_coords = wr_joyal.obsm["X_umap"]
txn1 = sc.get.obs_df(wr_joyal, keys=["TXN1"])["TXN1"].to_numpy()
txn1_low, txn1_high = txn1.min(), txn1.max()

# drawn with scatter rather than sc.pl.umap: it puts the whole embedding underneath in grey,
# and it fixes the point size, which scanpy scales by cell count — the sparse panels would
# otherwise carry much larger points than the dense ones.
fig, axes = plt.subplots(len(timepoints), len(condition_order), squeeze=False,
                         figsize=(5.6, 4.4), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
for row, timepoint in enumerate(timepoints):
    for column, condition in enumerate(condition_order):
        ax = axes[row][column]
        in_stratum = (
            (wr_joyal.obs["timepoint"] == timepoint) & (wr_joyal.obs["condition"] == condition)
        ).to_numpy()

        # every cell in grey, so each panel is read against the same outline rather than
        # appearing to be a differently-shaped dataset
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=1, c="#bfbfbf", linewidths=0)
        points = ax.scatter(umap_coords[in_stratum, 0], umap_coords[in_stratum, 1], s=1,
                            c=txn1[in_stratum], cmap="viridis",
                            vmin=txn1_low, vmax=txn1_high, linewidths=0)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_ylabel(f"Txn1 — {timepoint}" if column == 0 else "", fontsize=8)
        ax.set_title(condition if row == 0 else "", fontsize=9)
        ax.text(0.02, 0.98, f"n={in_stratum.sum():,}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

colorbar = fig.colorbar(points, ax=axes, location="right", shrink=0.35, pad=0.02, aspect=25)
colorbar_ticks = np.linspace(txn1_low, txn1_high, 3)
colorbar.set_ticks(colorbar_ticks, labels=[f"{tick:.2f}" for tick in colorbar_ticks])
colorbar.ax.tick_params(labelsize=7)

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/umap-txn1-stratified-wr-joyal-output-1.png"
id="umap-txn1-stratified-wr-joyal" />

## Txn1 by cell type across the design

``` python
cell_type_order = list(wr_joyal.obs["cell_type"].cat.categories)
condition_order = list(wr_joyal.obs["condition"].cat.categories)
timepoints = list(wr_joyal.obs["timepoint"].cat.categories)

# grouped violins rather than four panels: NORM and OIR sit side by side within each cell
# type's slot, so the comparison is a glance rather than a jump between figures. Timepoint is
# the facet, which leaves the interaction readable down a column.
fig, axes = plt.subplots(len(timepoints), 1, squeeze=False, sharex=True, sharey=True,
                         figsize=(9, 5.0), constrained_layout=True)
for row, (ax, timepoint) in enumerate(zip(axes.flat, timepoints)):
    at_timepoint = txn1_df[txn1_df["timepoint"] == timepoint]
    sns.violinplot(data=at_timepoint, x="cell_type", y="TXN1", order=cell_type_order,
                   hue="condition", hue_order=condition_order, palette=condition_palette,
                   gap=0.1, cut=0, density_norm="width", inner="box",
                   inner_kws=violin_inner_kws, linewidth=0.5, saturation=1,
                   legend=(row == 0), ax=ax)
    ax.set_ylabel(f"Txn1 — {timepoint}", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

axes[0][0].legend(title="", fontsize=7, frameon=False,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
axes[-1][0].tick_params(axis="x", labelrotation=45, labelsize=8)
for label in axes[-1][0].get_xticklabels():
    label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-stratified-wr-joyal-output-1.png"
id="txn1-violin-stratified-wr-joyal" />

## Txn1 by cell type across the design, grouped by timepoint

``` python
timepoint_palette = dict(zip(wr_joyal.obs["timepoint"].cat.categories,
                             wr_joyal.uns["timepoint_colors"]))

# the same data with the roles swapped: P14 and P17 side by side within each cell type, and
# condition as the facet. Reading down a column here compares the two timepoints within a
# condition, where the plot above compares the two conditions within a timepoint.
fig, axes = plt.subplots(len(condition_order), 1, squeeze=False, sharex=True, sharey=True,
                         figsize=(9, 5.0), constrained_layout=True)
for row, (ax, condition) in enumerate(zip(axes.flat, condition_order)):
    in_condition = txn1_df[txn1_df["condition"] == condition]
    sns.violinplot(data=in_condition, x="cell_type", y="TXN1", order=cell_type_order,
                   hue="timepoint", hue_order=timepoints, palette=timepoint_palette,
                   gap=0.1, cut=0, density_norm="width", inner="box",
                   inner_kws=violin_inner_kws, linewidth=0.5, saturation=1,
                   legend=(row == 0), ax=ax)
    ax.set_ylabel(f"Txn1 — {condition}", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

axes[0][0].legend(title="", fontsize=7, frameon=False,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
axes[-1][0].tick_params(axis="x", labelrotation=45, labelsize=8)
for label in axes[-1][0].get_xticklabels():
    label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-stratified-by-condition-wr-joyal-output-1.png"
id="txn1-violin-stratified-by-condition-wr-joyal" />

# Rod-depleted Retina

## Cluster Cells

``` python
cd73ft_joyal_resolution = 0.50   # chosen in oir-analysis and oir-analysis-2
leiden_key = f"leiden_res_{cd73ft_joyal_resolution:.2f}_v0"
ranked_key = f"leiden_res_{cd73ft_joyal_resolution:.2f}_v1"

cd73ft_joyal_clustered_path = Path("data/processed/GSE150703_adata_Cd73ft_Joyal_clustered.h5ad")

if cd73ft_joyal_clustered_path.exists():        # clustered once, then read back on later renders
    cd73ft_joyal = sc.read_h5ad(cd73ft_joyal_clustered_path)
else:
    cd73ft_joyal = adata[adata.obs["batch"] == "Cd73ft_Joyal"].copy()   # this batch only

    for column in ["condition", "timepoint", "prep", "lab", "replicate", "batch"]:
        cd73ft_joyal.obs[column] = cd73ft_joyal.obs[column].cat.remove_unused_categories()  # drop empty levels

    cd73ft_joyal.var["mt"] = cd73ft_joyal.var_names.str.startswith("MT-")
    cd73ft_joyal.var["ribo"] = cd73ft_joyal.var_names.str.startswith(("RPS", "RPL"))
    cd73ft_joyal.var["hb"] = cd73ft_joyal.var_names.str.startswith(("HBA-", "HBB-"))
    sc.pp.calculate_qc_metrics(
        cd73ft_joyal,
        qc_vars=["mt", "ribo", "hb"],
        expr_type="log1p",                     # X is normalized, so these are not UMI fractions
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    sc.pp.filter_genes(cd73ft_joyal, min_cells=3)  # rarely detected genes distort the variable set
    sc.pp.highly_variable_genes(cd73ft_joyal, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(cd73ft_joyal, svd_solver="arpack")
    sc.pp.neighbors(cd73ft_joyal, n_neighbors=10, n_pcs=40)
    sc.tl.umap(cd73ft_joyal)

    sc.tl.leiden(
        cd73ft_joyal,
        resolution=cd73ft_joyal_resolution,
        key_added=leiden_key,
        flavor="igraph",
        random_state=0,                        # the one argument here that decides the output
    )

    # renumbered by size, so cluster 1 is always the largest
    cluster_sizes = cd73ft_joyal.obs[leiden_key].value_counts()
    ranked_labels = [str(rank) for rank in range(1, len(cluster_sizes) + 1)]
    cd73ft_joyal.obs[ranked_key] = (
        cd73ft_joyal.obs[leiden_key]
        .map(dict(zip(cluster_sizes.index, ranked_labels)))
        .astype("category")
        .cat.reorder_categories(ranked_labels)
    )

    for obs_key, palette in obs_palettes.items():
        categories = cd73ft_joyal.obs[obs_key].cat.categories
        cd73ft_joyal.uns[f"{obs_key}_colors"] = [palette[category] for category in categories]

    cd73ft_joyal.uns[f"{ranked_key}_colors"] = [               # tab20, cycled once past 20 clusters
        mcolors.to_hex(plt.cm.tab20.colors[(int(label) - 1) % 20]) for label in ranked_labels
    ]

    cd73ft_joyal.write_h5ad(cd73ft_joyal_clustered_path)

cd73ft_joyal
```

    AnnData object with n_obs × n_vars = 10329 × 18098
        obs: 'condition', 'timepoint', 'prep', 'lab', 'replicate', 'batch', 'n_genes_by_log1p', 'total_log1p', 'total_log1p_mt', 'pct_log1p_mt', 'total_log1p_ribo', 'pct_log1p_ribo', 'total_log1p_hb', 'pct_log1p_hb', 'leiden_res_0.50_v0', 'leiden_res_0.50_v1'
        var: 'mt', 'ribo', 'hb', 'n_cells_by_log1p', 'mean_log1p', 'pct_dropout_by_log1p', 'total_log1p', 'n_cells', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
        uns: 'batch_colors', 'condition_colors', 'hvg', 'lab_colors', 'leiden_res_0.50_v0', 'leiden_res_0.50_v1_colors', 'neighbors', 'pca', 'prep_colors', 'timepoint_colors', 'umap'
        obsm: 'X_pca', 'X_umap'
        varm: 'PCs'
        obsp: 'connectivities', 'distances'
        layers: None (.X)

## Clusters on the UMAP

``` python
# scanpy builds the figure here, and takes no figsize argument, so rcParams is where its size
# has to be set
plt.rcParams["figure.figsize"] = (5.0, 4.3)

ax = sc.pl.umap(
    cd73ft_joyal,
    color=ranked_key,
    legend_loc="on data",
    legend_fontoutline=2,
    frameon=True,
    show=False,
)
ax.set_aspect("equal", adjustable="datalim")
plt.show()
```

<img
src="oir_analysis_files/figure-commonmark/umap-clusters-cd73ft-joyal-output-1.png"
id="umap-clusters-cd73ft-joyal" />

## QC metrics

``` python
qc_metrics = {
    "n_genes_by_log1p": "Genes detected",
    "total_log1p": "Total expression",
    "pct_log1p_mt": "% mitochondrial",
    "pct_log1p_ribo": "% ribosomal",
    "pct_log1p_hb": "% hemoglobin",
}
violin_inner_kws = {"box_width": 2, "whis_width": 1, "marker": "o",
                    "markersize": 3.0, "markeredgecolor": "white"}

qc_df = sc.get.obs_df(cd73ft_joyal, keys=[ranked_key] + list(qc_metrics))
batch_color = cd73ft_joyal.uns["batch_colors"][0]

fig, axes = plt.subplots(1, len(qc_metrics), squeeze=False, figsize=(9, 3.0),
                         constrained_layout=True)
for ax, (qc_key, qc_label) in zip(axes.flat, qc_metrics.items()):
    sns.violinplot(data=qc_df, y=qc_key, color=batch_color, cut=0, density_norm="width",
                   inner="box", inner_kws=violin_inner_kws, linewidth=0.5, ax=ax)
    ax.set_ylabel(qc_label, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/qc-violin-cd73ft-joyal-output-1.png"
id="qc-violin-cd73ft-joyal" />

## QC metrics by cluster

``` python
# only scanpy reads the _colors entry out of uns, so the palette is handed to seaborn
# explicitly — otherwise a cluster changes color between the UMAP and these violins
cluster_palette = dict(zip(cd73ft_joyal.obs[ranked_key].cat.categories, cd73ft_joyal.uns[f"{ranked_key}_colors"]))

fig, axes = plt.subplots(len(qc_metrics), 1, squeeze=False, sharex=True, figsize=(9, 4.3),
                         constrained_layout=True)
for ax, (qc_key, qc_label) in zip(axes.flat, qc_metrics.items()):
    # dodge=False because hue repeats x: left on, seaborn gives each of the 21 clusters its
    # own sub-slot and every violin is drawn a twentieth of its slot wide, off its tick
    sns.violinplot(data=qc_df, x=ranked_key, y=qc_key, hue=ranked_key, palette=cluster_palette,
                   legend=False, dodge=False, cut=0, density_norm="width", inner="box",
                   inner_kws=violin_inner_kws, linewidth=0.5, saturation=1, ax=ax)

    # seaborn colours the bodies in an order that does not follow the x positions when hue
    # repeats x, so each is recoloured from the cluster it actually sits under. The order
    # comes off the categorical rather than the tick labels, which are empty on a shared axis.
    cluster_order = list(cd73ft_joyal.obs[ranked_key].cat.categories)
    for body in ax.collections:
        extents = body.get_paths()[0].get_extents()
        body.set_facecolor(cluster_palette[cluster_order[round((extents.x0 + extents.x1) / 2)]])

    ax.set_ylabel(qc_label, fontsize=7)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)

axes[-1][0].set_xlabel("Cluster", fontsize=9)
axes[-1][0].tick_params(axis="x", labelsize=8)
plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/qc-violin-by-cluster-cd73ft-joyal-output-1.png"
id="qc-violin-by-cluster-cd73ft-joyal" />

## Correlation with the reference

``` python
# correlated over the query's variable genes rather than everything shared: the
# housekeeping bulk is near-identical in every class and only lifts all twelve correlations
# together, flattening the differences the call is made on
feature_genes = [
    gene for gene in cd73ft_joyal.var_names[cd73ft_joyal.var["highly_variable"]]
    if gene in reference_centroids.var_names
]

cluster_profiles = pd.DataFrame(
    [
        np.asarray(
            cd73ft_joyal[cd73ft_joyal.obs[ranked_key] == cluster, feature_genes].X.mean(axis=0)
        ).ravel()
        for cluster in cd73ft_joyal.obs[ranked_key].cat.categories
    ],
    index=cd73ft_joyal.obs[ranked_key].cat.categories,
    columns=feature_genes,
)
reference_profiles = pd.DataFrame(
    reference_centroids[:, feature_genes].X,
    index=reference_centroids.obs_names,
    columns=feature_genes,
)

# Spearman, so a class is matched on the order it puts the genes in rather than on absolute
# levels — the two datasets were normalized by different pipelines. Ranking the rows and
# taking Pearson is the same thing, and does all 21 x 12 pairs in one call.
ranked_clusters = cluster_profiles.rank(axis=1)
ranked_reference = reference_profiles.rank(axis=1)
reference_correlation = pd.DataFrame(
    np.corrcoef(ranked_clusters, ranked_reference)[:len(ranked_clusters), len(ranked_clusters):],
    index=cluster_profiles.index,
    columns=reference_profiles.index,
)

print(f"{len(feature_genes)} of {int(cd73ft_joyal.var['highly_variable'].sum())} "
      f"variable genes found in the reference")
reference_correlation.round(3)
```

    1737 of 2000 variable genes found in the reference

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | AC | Astrocyte | BC | Cone | Endothelial | HC | MG | Microglia | Pericyte | RGC | RPE | Rod |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1 | 0.614 | 0.657 | 0.709 | 0.643 | 0.606 | 0.634 | 0.691 | 0.533 | 0.653 | 0.586 | 0.583 | 0.661 |
| 2 | 0.542 | 0.748 | 0.538 | 0.496 | 0.639 | 0.548 | 0.840 | 0.526 | 0.672 | 0.516 | 0.586 | 0.544 |
| 3 | 0.679 | 0.610 | 0.820 | 0.660 | 0.536 | 0.655 | 0.662 | 0.441 | 0.606 | 0.634 | 0.556 | 0.670 |
| 4 | 0.590 | 0.647 | 0.699 | 0.636 | 0.576 | 0.636 | 0.685 | 0.513 | 0.630 | 0.568 | 0.566 | 0.651 |
| 5 | 0.826 | 0.631 | 0.651 | 0.559 | 0.496 | 0.700 | 0.636 | 0.458 | 0.577 | 0.773 | 0.494 | 0.606 |
| 6 | 0.513 | 0.594 | 0.620 | 0.793 | 0.507 | 0.556 | 0.637 | 0.477 | 0.551 | 0.485 | 0.631 | 0.728 |
| 7 | 0.211 | 0.510 | 0.261 | 0.260 | 0.547 | 0.282 | 0.388 | 0.780 | 0.470 | 0.223 | 0.345 | 0.271 |
| 8 | 0.375 | 0.612 | 0.411 | 0.420 | 0.709 | 0.430 | 0.602 | 0.529 | 0.752 | 0.379 | 0.522 | 0.443 |
| 9 | 0.365 | 0.584 | 0.405 | 0.410 | 0.798 | 0.412 | 0.566 | 0.534 | 0.677 | 0.358 | 0.480 | 0.422 |
| 10 | 0.442 | 0.681 | 0.445 | 0.437 | 0.602 | 0.446 | 0.713 | 0.515 | 0.610 | 0.416 | 0.590 | 0.464 |
| 11 | 0.621 | 0.612 | 0.595 | 0.530 | 0.532 | 0.738 | 0.593 | 0.493 | 0.581 | 0.603 | 0.503 | 0.558 |
| 12 | 0.438 | 0.766 | 0.449 | 0.409 | 0.614 | 0.466 | 0.701 | 0.514 | 0.620 | 0.421 | 0.538 | 0.446 |
| 13 | 0.382 | 0.457 | 0.458 | 0.564 | 0.379 | 0.426 | 0.495 | 0.385 | 0.404 | 0.339 | 0.534 | 0.611 |
| 14 | 0.306 | 0.374 | 0.295 | 0.264 | 0.345 | 0.290 | 0.392 | 0.322 | 0.348 | 0.279 | 0.297 | 0.285 |

</div>

## Correlation with the reference by cluster

``` python
# the same matrix three ways, matching the marker heatmap. The raw correlations sit in a
# narrow band, so the two scaled panels are what can actually be read: down a column asks
# "which cluster is most this class", and across a row asks "which class best fits this
# cluster" — the annotation question, and the one the calls are taken from.
#
# only the row scaling is a call. A column's brightest cell is the best home for that class
# whether or not the class is present at all: RPE peaks on a rod cluster here, which says
# where it would land rather than that it was found.
class_range = reference_correlation.max(axis=0) - reference_correlation.min(axis=0)
scaled_by_class = reference_correlation.sub(
    reference_correlation.min(axis=0), axis=1
).div(class_range, axis=1)

correlation_range = reference_correlation.max(axis=1) - reference_correlation.min(axis=1)
scaled_correlation = reference_correlation.sub(
    reference_correlation.min(axis=1), axis=0
).div(correlation_range, axis=0)

correlation_panels = {
    "Spearman correlation": reference_correlation,
    "Scaled per class": scaled_by_class,
    "Scaled per cluster": scaled_correlation,
}

# walk the clusters in order and take each one's best class; a class already placed is
# skipped, and any class no cluster leads with is appended. The matches then read as a
# diagonal rather than being scattered across the panel.
class_order = []
for cluster in scaled_correlation.index:
    best = scaled_correlation.loc[cluster].idxmax()
    if best not in class_order:
        class_order.append(best)
class_order += [name for name in scaled_correlation.columns if name not in class_order]

fig, axes = plt.subplots(1, len(correlation_panels), squeeze=False, figsize=(9, 4.3),
                         constrained_layout=True)
for ax, (title, matrix) in zip(axes.flat, correlation_panels.items()):
    # xticklabels/yticklabels left at "auto" thins the labels to every other one once a
    # third panel is on the figure, which leaves half the classes unnamed
    sns.heatmap(matrix[class_order], cmap="viridis", linewidths=0.5, linecolor="white",
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "pad": 0.02}, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Reference class", fontsize=8)
    ax.set_ylabel("Cluster", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/heatmap-reference-cd73ft-joyal-output-1.png"
id="heatmap-reference-cd73ft-joyal" />

## Correlation with the reference by cell

Correlating each cell against the same centroids says which clusters
hold more than one cell type, without needing a marker panel to name the
second one.

``` python
# every shared gene here, where the cluster-level correlation above uses variable genes only.
# A centroid is dense, so its housekeeping bulk lifts all twelve classes together and flattens
# the differences; a single cell is 97% zero over those same variable genes and needs every
# gene it can get. Widening to all of them moves agreement with the cluster call from 90% to 95%.
shared_genes = [gene for gene in cd73ft_joyal.var_names if gene in reference_centroids.var_names]

ranked_cells = pd.DataFrame(np.asarray(cd73ft_joyal[:, shared_genes].X.todense())).rank(axis=1).to_numpy()
ranked_classes = pd.DataFrame(
    reference_centroids[:, shared_genes].X, index=reference_centroids.obs_names
).rank(axis=1).to_numpy()

# Spearman again, written out rather than looped so all twelve classes come out of one product
ranked_cells = ranked_cells - ranked_cells.mean(axis=1, keepdims=True)
ranked_classes = ranked_classes - ranked_classes.mean(axis=1, keepdims=True)
cell_correlation = (ranked_cells @ ranked_classes.T) / np.sqrt(
    (ranked_cells ** 2).sum(axis=1)[:, None] * (ranked_classes ** 2).sum(axis=1)[None, :]
)

cd73ft_joyal.obs["cell_call"] = pd.Categorical(
    reference_centroids.obs_names[cell_correlation.argmax(axis=1)],
    categories=sorted(reference_centroids.obs_names),
)

cell_composition = pd.crosstab(cd73ft_joyal.obs[ranked_key], cd73ft_joyal.obs["cell_call"])
(cell_composition.loc[:, cell_composition.sum() > 0]
 .div(cell_composition.sum(axis=1), axis=0).mul(100).round().astype(int))
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

| cell_call | AC | Astrocyte | BC | Cone | Endothelial | HC | MG | Microglia | Pericyte | RGC | RPE | Rod |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| leiden_res_0.50_v1 |  |  |  |  |  |  |  |  |  |  |  |  |
| 1 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 0 | 0 | 98 | 0 | 0 | 0 | 0 | 0 |
| 3 | 3 | 0 | 95 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4 | 1 | 0 | 98 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 72 | 0 | 6 | 0 | 0 | 1 | 1 | 0 | 0 | 18 | 0 | 0 |
| 6 | 0 | 0 | 3 | 96 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 7 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 97 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 99 | 0 | 0 | 0 |
| 9 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | 1 | 5 | 0 | 0 | 0 | 0 | 78 | 1 | 0 | 0 | 15 | 0 |
| 11 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | 2 | 0 | 9 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 84 |
| 14 | 7 | 5 | 49 | 2 | 19 | 0 | 14 | 0 | 0 | 5 | 0 | 0 |

</div>

## Marker scores

``` python
retina_markers = {
    "Rods": ["Rho", "Pde6b", "Nrl", "Nr2e3"],
    "Cones": ["Opn1mw", "Opn1sw", "Arr3", "Thrb"],
    "RGC": ["Rbpms", "Sncg", "Pou4f1", "Pou4f2"],
    "Amacrine": ["Tfap2a", "Tfap2b", "Chat", "Vsx1"],
    "Bipolar": ["Vsx2", "Isl1", "Prkca", "Cabp5"],
    "Horizontal": ["Onecut1", "Onecut2", "Lhx1", "Prox1"],
    "Muller_Glia": ["Rlbp1", "Glul", "Sox9", "Slc1a3"],
    "Microglia": ["Cx3cr1", "P2ry12", "Tmem119", "Aif1"],
    "Vascular_Endothelial": ["Cdh5", "Pecam1", "Tek"],
    "Pericytes": ["Pdgfrb", "Acta2", "Des"],
    "Astrocytes": ["Gfap", "Aldh1l1", "Aqp4"],
}

score_keys = []
markers_present = {}
for cell_type, genes in retina_markers.items():
    present = [gene.upper() for gene in genes if gene.upper() in cd73ft_joyal.var_names]
    score_key = f"score_{cell_type}"
    sc.tl.score_genes(cd73ft_joyal, gene_list=present, score_name=score_key)
    score_keys.append(score_key)
    markers_present[cell_type] = present
    print(f"{cell_type}: {len(present)}/{len(genes)} markers")
```

    Rods: 4/4 markers
    Cones: 4/4 markers
    RGC: 4/4 markers
    Amacrine: 4/4 markers
    Bipolar: 4/4 markers
    Horizontal: 4/4 markers
    Muller_Glia: 4/4 markers
    Microglia: 4/4 markers
    Vascular_Endothelial: 3/3 markers
    Pericytes: 3/3 markers
    Astrocytes: 3/3 markers

## Marker scores by cluster

``` python
score_df = sc.get.obs_df(cd73ft_joyal, keys=[ranked_key] + score_keys)

for start in range(0, len(score_keys), 6):
    group = score_keys[start:start + 6]
    group_types = list(retina_markers)[start:start + 6]

    fig, axes = plt.subplots(len(group), 1, squeeze=False, sharex=True, figsize=(9, 4.3),
                             constrained_layout=True)
    for ax, score_key, cell_type in zip(axes.flat, group, group_types):
        sns.violinplot(data=score_df, x=ranked_key, y=score_key, hue=ranked_key,
                       palette=cluster_palette, legend=False, dodge=False, cut=0,
                       density_norm="width", inner="box", inner_kws=violin_inner_kws,
                       linewidth=0.5, saturation=1, ax=ax)

        cluster_order = list(cd73ft_joyal.obs[ranked_key].cat.categories)
        for body in ax.collections:
            extents = body.get_paths()[0].get_extents()
            body.set_facecolor(cluster_palette[cluster_order[round((extents.x0 + extents.x1) / 2)]])

        ax.set_ylabel(f"{cell_type.replace('_', ' ')} score", fontsize=7)
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[-1][0].set_xlabel("Cluster", fontsize=9)
    axes[-1][0].tick_params(axis="x", labelsize=8)
    plt.show()
    plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/score-violin-by-cluster-cd73ft-joyal-output-1.png"
id="score-violin-by-cluster-cd73ft-joyal-1" />

<img
src="oir_analysis_files/figure-commonmark/score-violin-by-cluster-cd73ft-joyal-output-2.png"
id="score-violin-by-cluster-cd73ft-joyal-2" />

## Marker scores on the UMAP

``` python
for start in range(0, len(score_keys), 3):
    group = score_keys[start:start + 3]

    fig, axs = plt.subplots(2, 3, height_ratios=[1, 0.22], figsize=(8.6, 3.4),
                            constrained_layout=True)
    for ax, bar_ax, score_key in zip(axs[0], axs[1], group):
        sc.pl.umap(
            cd73ft_joyal,
            color=score_key,
            ax=ax,
            frameon=True,
            colorbar_loc=None,
            show=False,
        )
        ax.set_aspect("equal", adjustable="datalim")

        # the strip reserves the space; the colorbar drawn inside it is excluded from the
        # layout so its tick labels cannot push the panel around
        bar_ax.axis("off")
        colorbar_ax = bar_ax.inset_axes([0.325, 0.82, 0.35, 0.18])
        colorbar_ax.set_in_layout(False)
        colorbar = fig.colorbar(ax.collections[0], cax=colorbar_ax, orientation="horizontal")

        # placed explicitly at the ends and middle: the automatic locator picks its own round
        # numbers per score, so the tick count and positions differ from panel to panel
        score_low, score_high = ax.collections[0].get_clim()
        colorbar_ticks = np.linspace(score_low, score_high, 3)
        colorbar.set_ticks(colorbar_ticks, labels=[f"{tick:.2f}" for tick in colorbar_ticks])
        colorbar.ax.tick_params(labelsize=7)

    for empty_ax in axs[0][len(group):]:
        fig.delaxes(empty_ax)
    for empty_bar_ax in axs[1][len(group):]:
        empty_bar_ax.axis("off")

    plt.show()
    plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/umap-scores-cd73ft-joyal-output-1.png"
id="umap-scores-cd73ft-joyal-1" />

<img
src="oir_analysis_files/figure-commonmark/umap-scores-cd73ft-joyal-output-2.png"
id="umap-scores-cd73ft-joyal-2" />

<img
src="oir_analysis_files/figure-commonmark/umap-scores-cd73ft-joyal-output-3.png"
id="umap-scores-cd73ft-joyal-3" />

<img
src="oir_analysis_files/figure-commonmark/umap-scores-cd73ft-joyal-output-4.png"
id="umap-scores-cd73ft-joyal-4" />

## Marker genes by cluster

``` python
# figsize is deliberately not passed: dotplot derives its own from the gene and cluster counts,
# and forcing it to slide dimensions packs 41 genes across 21 clusters until the dots overlap
sc.pl.dotplot(cd73ft_joyal, markers_present, groupby=ranked_key, standard_scale="var")
```

<img
src="oir_analysis_files/figure-commonmark/dotplot-markers-cd73ft-joyal-output-1.png"
id="dotplot-markers-cd73ft-joyal" />

## Marker score heatmap

``` python
mean_scores = score_df.groupby(ranked_key, observed=True)[score_keys].mean()
mean_scores.columns = [cell_type.replace("_", " ") for cell_type in retina_markers]

# the same matrix three ways, because they answer different questions: the raw means show
# magnitude, scaling down a column asks "which cluster is most this cell type", and scaling
# across a row asks "which cell type best fits this cluster" — the annotation question
scaled_by_type = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min())
row_range = mean_scores.max(axis=1) - mean_scores.min(axis=1)
scaled_by_cluster = mean_scores.sub(mean_scores.min(axis=1), axis=0).div(row_range, axis=0)

score_normalizations = {
    "Mean score": mean_scores,
    "Scaled per cell type": scaled_by_type,
    "Scaled per cluster": scaled_by_cluster,
}

# walk the clusters in order and take each one's best cell type; a type already placed is
# skipped, and any type no cluster leads with is appended. The matches then read as a
# diagonal rather than being scattered across the panel.
column_order = []
for cluster in scaled_by_cluster.index:
    best = scaled_by_cluster.loc[cluster].idxmax()
    if best not in column_order:
        column_order.append(best)
column_order += [c for c in scaled_by_cluster.columns if c not in column_order]

fig, axes = plt.subplots(1, len(score_normalizations), squeeze=False, figsize=(9, 4.3),
                         constrained_layout=True)
for ax, (title, matrix) in zip(axes.flat, score_normalizations.items()):
    # as on the reference heatmap: left at "auto" the labels thin to every other one across
    # three panels, which leaves half the cell types and half the clusters unnamed
    sns.heatmap(matrix[column_order], cmap="viridis", linewidths=0.5, linecolor="white",
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "pad": 0.02}, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Cell type", fontsize=8)
    ax.set_ylabel("Cluster", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/heatmap-scores-cd73ft-joyal-output-1.png"
id="heatmap-scores-cd73ft-joyal" />

## Preliminary cell types

``` python
# the call is the best-correlating reference class. The marker scores stay in the document
# as an independent read on the same question — where the two disagree, the disagreement is
# the thing to look at before writing an override.
cluster_calls = reference_correlation.idxmax(axis=1)
marker_calls = mean_scores.idxmax(axis=1)

# no calls made by hand yet, so every cluster below is the reference argmax
cluster_call_overrides = {}
cluster_calls.update(pd.Series(cluster_call_overrides))

cd73ft_joyal.obs["cell_type"] = cd73ft_joyal.obs[ranked_key].map(cluster_calls).astype("category")

# keyed on every class in the reference rather than only the ones called, so a cell type
# keeps its colour whatever a given resolution happens to find
cell_type_palette = dict(zip(sorted(reference_centroids.obs_names), sc.pl.palettes.default_20))
cd73ft_joyal.uns["cell_type_colors"] = [
    cell_type_palette[cell_type] for cell_type in cd73ft_joyal.obs["cell_type"].cat.categories
]

# margin is the gap to the runner-up class: a cluster called on a hair's difference is one
# to read the heatmap for, however high its correlation
sorted_correlation = np.sort(reference_correlation.to_numpy(), axis=1)

pd.DataFrame({
    "cell_type": cluster_calls,
    "cells": cd73ft_joyal.obs[ranked_key].value_counts(),
    "correlation": reference_correlation.max(axis=1).round(3),
    "margin": (sorted_correlation[:, -1] - sorted_correlation[:, -2]).round(3),
    "marker call": marker_calls,
})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | cell_type   | cells | correlation | margin | marker call          |
|-----|-------------|-------|-------------|--------|----------------------|
| 1   | BC          | 2860  | 0.709       | 0.019  | Bipolar              |
| 2   | MG          | 1899  | 0.840       | 0.092  | Muller Glia          |
| 3   | BC          | 1287  | 0.820       | 0.141  | Bipolar              |
| 4   | BC          | 1244  | 0.699       | 0.014  | Bipolar              |
| 5   | AC          | 916   | 0.826       | 0.053  | Amacrine             |
| 6   | Cone        | 742   | 0.793       | 0.064  | Cones                |
| 7   | Microglia   | 305   | 0.780       | 0.233  | Microglia            |
| 8   | Pericyte    | 305   | 0.752       | 0.044  | Pericytes            |
| 9   | Endothelial | 218   | 0.798       | 0.121  | Vascular Endothelial |
| 10  | MG          | 197   | 0.713       | 0.031  | Muller Glia          |
| 11  | HC          | 120   | 0.738       | 0.117  | Horizontal           |
| 12  | Astrocyte   | 113   | 0.766       | 0.065  | Muller Glia          |
| 13  | Rod         | 80    | 0.611       | 0.047  | Rods                 |
| 14  | MG          | 43    | 0.392       | 0.018  | Muller Glia          |

</div>

## Refining a call with the per-cell correlations

``` python
# the reference calls a whole cluster one class, so a population that never forms its own
# cluster cannot be named from the cluster call alone. The per-cell correlations above say
# where that happened: a class a real share of a cluster's cells correlate best with is a
# population inside it, and those cells take that class instead of the cluster's.
#
# a share rather than a count, because a handful of cells favouring a class is what dropout
# does to every cluster; and a count as well, so a small cluster cannot qualify a class on
# three cells. RPE takes 14% of one cluster here and is excluded by the share, which is the
# right answer for a class with no business in a neural retina prep.
min_share = 0.25
min_cells = 10

refinements = []
for cluster in cell_composition.index:
    counts = cell_composition.loc[cluster]
    populations = counts[(counts >= min_cells) & (counts / counts.sum() >= min_share)]
    in_cluster = (cd73ft_joyal.obs[ranked_key] == cluster).to_numpy()

    for population in populations.index:
        if population not in cd73ft_joyal.obs["cell_type"].cat.categories:
            cd73ft_joyal.obs["cell_type"] = cd73ft_joyal.obs["cell_type"].cat.add_categories([population])
        # a cell keeps the cluster's call unless its own best class is one that qualified
        takes = in_cluster & (cd73ft_joyal.obs["cell_call"] == population).to_numpy()
        cd73ft_joyal.obs.loc[takes, "cell_type"] = population

    refinements.append({
        "cluster": cluster,
        "call": cluster_calls[cluster],
        "cells": int(in_cluster.sum()),
        "populations": ", ".join(f"{name} {n}" for name, n in populations.items()),
        "refined": bool(len(populations) > 1 or populations.index[0] != cluster_calls[cluster]),
    })

# a cluster with one qualifying population that is already its call is left exactly as it was
print(pd.DataFrame(refinements).to_string(index=False))

# a class added above lands at the end of the categories, which would put it last on every
# axis downstream rather than in with the rest
cd73ft_joyal.obs["cell_type"] = cd73ft_joyal.obs["cell_type"].cat.remove_unused_categories()
cd73ft_joyal.obs["cell_type"] = cd73ft_joyal.obs["cell_type"].cat.reorder_categories(
    sorted(cd73ft_joyal.obs["cell_type"].cat.categories)
)
cd73ft_joyal.uns["cell_type_colors"] = [
    cell_type_palette[cell_type] for cell_type in cd73ft_joyal.obs["cell_type"].cat.categories
]
```

    cluster        call  cells     populations  refined
          1          BC   2860         BC 2842    False
          2          MG   1899         MG 1869    False
          3          BC   1287         BC 1228    False
          4          BC   1244         BC 1225    False
          5          AC    916          AC 662    False
          6        Cone    742        Cone 710    False
          7   Microglia    305   Microglia 296    False
          8    Pericyte    305    Pericyte 303    False
          9 Endothelial    218 Endothelial 218    False
         10          MG    197          MG 154    False
         11          HC    120          HC 120    False
         12   Astrocyte    113   Astrocyte 113    False
         13         Rod     80          Rod 67    False
         14          MG     43           BC 21     True

## Cluster composition by cell

``` python
# every cluster's per-cell composition as one bar. A cluster that is one cell type is a single
# block; a mixed one is divided, and the dashed line is the share a class has to clear to be
# treated as a population rather than as dropout.
shares = cell_composition.div(cell_composition.sum(axis=1), axis=0)
shares = shares.loc[:, shares.max() > 0.01]

fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
left = np.zeros(len(shares))
for population in shares.columns:
    ax.barh(shares.index.astype(str), shares[population], left=left,
            color=cell_type_palette[population], label=population, height=0.8)
    left += shares[population].to_numpy()

ax.axvline(min_share, color="#404040", linewidth=1.0, linestyle="--", zorder=3)
ax.set_xlabel("Share of the cluster's cells", fontsize=9)
ax.set_ylabel("Cluster", fontsize=9)
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.tick_params(labelsize=8)
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/refine-scatter-cd73ft-joyal-output-1.png"
id="refine-scatter-cd73ft-joyal" />

## Preliminary cell types on the UMAP

``` python
plt.rcParams["figure.figsize"] = (6.5, 4.3)

ax = sc.pl.umap(
    cd73ft_joyal,
    color="cell_type",
    frameon=True,
    show=False,
)
ax.set_aspect("equal", adjustable="datalim")
plt.show()
```

<img
src="oir_analysis_files/figure-commonmark/umap-cell-types-cd73ft-joyal-output-1.png"
id="umap-cell-types-cd73ft-joyal" />

## Txn1 on the UMAP

``` python
fig, axs = plt.subplots(2, 1, height_ratios=[1, 0.22], figsize=(4.3, 3.9),
                        constrained_layout=True)

sc.pl.umap(cd73ft_joyal, color="TXN1", ax=axs[0], frameon=True, colorbar_loc=None, show=False)
axs[0].set_aspect("equal", adjustable="datalim")

axs[1].axis("off")
colorbar_ax = axs[1].inset_axes([0.325, 0.82, 0.35, 0.18])
colorbar_ax.set_in_layout(False)
colorbar = fig.colorbar(axs[0].collections[0], cax=colorbar_ax, orientation="horizontal")

txn1_low, txn1_high = axs[0].collections[0].get_clim()
colorbar_ticks = np.linspace(txn1_low, txn1_high, 3)
colorbar.set_ticks(colorbar_ticks, labels=[f"{tick:.2f}" for tick in colorbar_ticks])
colorbar.ax.tick_params(labelsize=7)

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/umap-txn1-cd73ft-joyal-output-1.png"
id="umap-txn1-cd73ft-joyal" />

## Txn1 by cell type

``` python
txn1_df = sc.get.obs_df(cd73ft_joyal, keys=["TXN1", "cell_type", "condition", "timepoint"])

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
sns.violinplot(data=txn1_df, x="cell_type", y="TXN1", hue="cell_type",
               palette=cell_type_palette, legend=True, dodge=False, cut=0,
               density_norm="width", inner="box", inner_kws=violin_inner_kws,
               linewidth=0.5, saturation=1, ax=ax)

# with hue repeating x and dodge off, seaborn hands the bodies their colours in an order
# that does not follow the x positions — the legend ends up right and the violins wrong.
# Recolour each body from the cell type it actually sits under.
cell_type_order = list(cd73ft_joyal.obs["cell_type"].cat.categories)
for body in ax.collections:
    extents = body.get_paths()[0].get_extents()
    body.set_facecolor(cell_type_palette[cell_type_order[round((extents.x0 + extents.x1) / 2)]])

ax.set_xlabel("")
ax.set_ylabel("Txn1 (log1p)", fontsize=9)
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
ax.set_axisbelow(True)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-celltype-cd73ft-joyal-output-1.png"
id="txn1-violin-celltype-cd73ft-joyal" />

## Txn1 by cell type and condition

``` python
condition_palette = dict(zip(cd73ft_joyal.obs["condition"].cat.categories,
                             cd73ft_joyal.uns["condition_colors"]))

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
sns.violinplot(data=txn1_df, x="cell_type", y="TXN1", hue="condition",
               palette=condition_palette, cut=0, density_norm="width", inner="box",
               inner_kws=violin_inner_kws, linewidth=0.5, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Txn1 (log1p)", fontsize=9)
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
ax.set_axisbelow(True)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-condition-cd73ft-joyal-output-1.png"
id="txn1-violin-condition-cd73ft-joyal" />

## Txn1 by cell type and timepoint

``` python
timepoint_palette = dict(zip(cd73ft_joyal.obs["timepoint"].cat.categories,
                             cd73ft_joyal.uns["timepoint_colors"]))

fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
sns.violinplot(data=txn1_df, x="cell_type", y="TXN1", hue="timepoint",
               palette=timepoint_palette, cut=0, density_norm="width", inner="box",
               inner_kws=violin_inner_kws, linewidth=0.5, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Txn1 (log1p)", fontsize=9)
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
ax.set_axisbelow(True)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(title="", fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-timepoint-cd73ft-joyal-output-1.png"
id="txn1-violin-timepoint-cd73ft-joyal" />

## Txn1 across the design

``` python
# laid out like the violin slides below: timepoint down the rows, condition across the
# columns, so the same comparison sits in the same direction in every Txn1 figure
condition_order = list(cd73ft_joyal.obs["condition"].cat.categories)
timepoints = list(cd73ft_joyal.obs["timepoint"].cat.categories)

umap_coords = cd73ft_joyal.obsm["X_umap"]
txn1 = sc.get.obs_df(cd73ft_joyal, keys=["TXN1"])["TXN1"].to_numpy()
txn1_low, txn1_high = txn1.min(), txn1.max()

# drawn with scatter rather than sc.pl.umap: it puts the whole embedding underneath in grey,
# and it fixes the point size, which scanpy scales by cell count — the sparse panels would
# otherwise carry much larger points than the dense ones.
fig, axes = plt.subplots(len(timepoints), len(condition_order), squeeze=False,
                         figsize=(5.6, 4.4), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
for row, timepoint in enumerate(timepoints):
    for column, condition in enumerate(condition_order):
        ax = axes[row][column]
        in_stratum = (
            (cd73ft_joyal.obs["timepoint"] == timepoint) & (cd73ft_joyal.obs["condition"] == condition)
        ).to_numpy()

        # every cell in grey, so each panel is read against the same outline rather than
        # appearing to be a differently-shaped dataset
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=1, c="#bfbfbf", linewidths=0)
        points = ax.scatter(umap_coords[in_stratum, 0], umap_coords[in_stratum, 1], s=1,
                            c=txn1[in_stratum], cmap="viridis",
                            vmin=txn1_low, vmax=txn1_high, linewidths=0)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_ylabel(f"Txn1 — {timepoint}" if column == 0 else "", fontsize=8)
        ax.set_title(condition if row == 0 else "", fontsize=9)
        ax.text(0.02, 0.98, f"n={in_stratum.sum():,}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

colorbar = fig.colorbar(points, ax=axes, location="right", shrink=0.35, pad=0.02, aspect=25)
colorbar_ticks = np.linspace(txn1_low, txn1_high, 3)
colorbar.set_ticks(colorbar_ticks, labels=[f"{tick:.2f}" for tick in colorbar_ticks])
colorbar.ax.tick_params(labelsize=7)

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/umap-txn1-stratified-cd73ft-joyal-output-1.png"
id="umap-txn1-stratified-cd73ft-joyal" />

## Txn1 by cell type across the design

``` python
cell_type_order = list(cd73ft_joyal.obs["cell_type"].cat.categories)
condition_order = list(cd73ft_joyal.obs["condition"].cat.categories)
timepoints = list(cd73ft_joyal.obs["timepoint"].cat.categories)

# grouped violins rather than four panels: NORM and OIR sit side by side within each cell
# type's slot, so the comparison is a glance rather than a jump between figures. Timepoint is
# the facet, which leaves the interaction readable down a column.
fig, axes = plt.subplots(len(timepoints), 1, squeeze=False, sharex=True, sharey=True,
                         figsize=(9, 5.0), constrained_layout=True)
for row, (ax, timepoint) in enumerate(zip(axes.flat, timepoints)):
    at_timepoint = txn1_df[txn1_df["timepoint"] == timepoint]
    sns.violinplot(data=at_timepoint, x="cell_type", y="TXN1", order=cell_type_order,
                   hue="condition", hue_order=condition_order, palette=condition_palette,
                   gap=0.1, cut=0, density_norm="width", inner="box",
                   inner_kws=violin_inner_kws, linewidth=0.5, saturation=1,
                   legend=(row == 0), ax=ax)
    ax.set_ylabel(f"Txn1 — {timepoint}", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

axes[0][0].legend(title="", fontsize=7, frameon=False,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
axes[-1][0].tick_params(axis="x", labelrotation=45, labelsize=8)
for label in axes[-1][0].get_xticklabels():
    label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-stratified-cd73ft-joyal-output-1.png"
id="txn1-violin-stratified-cd73ft-joyal" />

## Txn1 by cell type across the design, grouped by timepoint

``` python
timepoint_palette = dict(zip(cd73ft_joyal.obs["timepoint"].cat.categories,
                             cd73ft_joyal.uns["timepoint_colors"]))

# the same data with the roles swapped: P14 and P17 side by side within each cell type, and
# condition as the facet. Reading down a column here compares the two timepoints within a
# condition, where the plot above compares the two conditions within a timepoint.
fig, axes = plt.subplots(len(condition_order), 1, squeeze=False, sharex=True, sharey=True,
                         figsize=(9, 5.0), constrained_layout=True)
for row, (ax, condition) in enumerate(zip(axes.flat, condition_order)):
    in_condition = txn1_df[txn1_df["condition"] == condition]
    sns.violinplot(data=in_condition, x="cell_type", y="TXN1", order=cell_type_order,
                   hue="timepoint", hue_order=timepoints, palette=timepoint_palette,
                   gap=0.1, cut=0, density_norm="width", inner="box",
                   inner_kws=violin_inner_kws, linewidth=0.5, saturation=1,
                   legend=(row == 0), ax=ax)
    ax.set_ylabel(f"Txn1 — {condition}", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", color="#b0b0b0", linewidth=0.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

axes[0][0].legend(title="", fontsize=7, frameon=False,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
axes[-1][0].tick_params(axis="x", labelrotation=45, labelsize=8)
for label in axes[-1][0].get_xticklabels():
    label.set_ha("right")

plt.show()
plt.close(fig)
```

<img
src="oir_analysis_files/figure-commonmark/txn1-violin-stratified-by-condition-cd73ft-joyal-output-1.png"
id="txn1-violin-stratified-by-condition-cd73ft-joyal" />
