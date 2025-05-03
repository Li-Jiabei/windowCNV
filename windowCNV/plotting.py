# plotting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.sparse import issparse

# --- Plotting: Heatmap of inferred CNV per cell ---
def plot_cna_heatmap(adata, chromosome, cell_type, layer="counts"):
    chr_mask = adata.var["chromosome"] == chromosome
    if chr_mask.sum() == 0:
        raise ValueError(f"No genes found on {chromosome}")

    cell_mask = adata.obs["cell_type"] == cell_type
    if cell_mask.sum() == 0:
        raise ValueError(f"No cells found for type {cell_type}")

    sub_adata = adata[cell_mask, :][:, chr_mask]
    X = sub_adata.layers[layer] if layer else sub_adata.X
    if issparse(X):
        X = X.toarray()

    plt.figure(figsize=(10, 0.2 * X.shape[0]))
    plt.imshow(X, aspect='auto', cmap='RdBu_r', interpolation='none')
    plt.title(f"{cell_type} on {chromosome}")
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.colorbar(label=layer)
    plt.tight_layout()
    plt.show()

# --- Plotting: Groundtruth + Inferred CNV map ---
def plot_groundtruth_and_inferred_cnv(
    adata,
    cell_type: str,
    celltype_key: str = "cell_type",
    cnv_truth_key: str = "simulated_cnvs",
    called_cnas_key: str = "called_cnas",
):
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    rows = []

    for chrom in chromosomes:
        truth = [
            x for x in adata.obs.loc[adata.obs[celltype_key] == cell_type, cnv_truth_key].explode()
            if x[0] == chrom
        ]
        inferred = [
            x for x in adata.obs.loc[adata.obs[celltype_key] == cell_type, called_cnas_key].explode()
            if x[0] == chrom
        ]
        for (s, e, t) in [(x[1], x[2], x[3]) for x in truth]:
            rows.append((chrom, s, e, t, "truth"))
        for (s, e, t) in [(x[1], x[2], x[3]) for x in inferred]:
            rows.append((chrom, s, e, t, "inferred"))

    df = pd.DataFrame(rows, columns=["chromosome", "start", "end", "type", "source"])
    df["color"] = df["type"].map({"gain": "red", "loss": "blue"})
    df = df.sort_values(by=["chromosome", "start"])

    plt.figure(figsize=(10, 0.4 * len(chromosomes)))
    for i, chrom in enumerate(chromosomes):
        for _, row in df[df["chromosome"] == chrom].iterrows():
            plt.plot([row["start"], row["end"]], [i, i], color=row["color"], linewidth=6, alpha=0.6 if row["source"] == "truth" else 1)
    plt.yticks(range(len(chromosomes)), chromosomes)
    plt.xlabel("Genomic Position")
    plt.title(f"CNV: {cell_type}")
    plt.tight_layout()
    plt.show()

# --- Plotting: Chromosome heatmap ---
def plot_cnv_groundtruth_vs_inferred(
    df: pd.DataFrame,
    metric: str = "f1",
    figsize: tuple = (10, 1)
):
    heatmap_df = df.set_index(["cell_type", "chromosome", "CN", "groundtruth"])[[metric]]
    plt.figure(figsize=(figsize[0], figsize[1] * len(heatmap_df)))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="black",
        mask=heatmap_df.isnull(),
        cbar_kws={'label': metric},
        vmin=0.0,
        vmax=1.0
    )
    plt.title(f"CNV Inference Metrics ({metric})")
    plt.tight_layout()
    plt.show()

# --- Plotting: Inferred CNV matrix ---
def plot_inferred_cnv_map(
    adata,
    cell_type: str,
    cnv_key: str = "cnv",
    vmin: float = -1,
    vmax: float = 1,
):
    cnv = adata.obsm[f"X_{cnv_key}"]
    chr_pos = adata.uns[cnv_key]["chr_pos"]
    if issparse(cnv):
        cnv = cnv.toarray()
    cell_mask = adata.obs["cell_type"] == cell_type
    X = cnv[cell_mask, :]

    plt.figure(figsize=(10, 0.2 * X.shape[0]))
    plt.imshow(X, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f"Inferred CNV map: {cell_type}")
    plt.colorbar(label="CNV signal")
    plt.tight_layout()
    plt.show()

# --- Evaluation: Groundtruth vs inferred ---
def evaluate_cnv_inference_aligned(adata, true_key="simulated_cnvs", pred_key="called_cnas", celltype_key="cell_type"):
    rows = []
    for ct in adata.obs[celltype_key].unique():
        true = adata.obs.loc[adata.obs[celltype_key] == ct, true_key].explode()
        pred = adata.obs.loc[adata.obs[celltype_key] == ct, pred_key].explode()

        gt_events = {(x[0], x[1], x[2], x[3]) for x in true}
        pr_events = {(x[0], x[1], x[2], x[3]) for x in pred}

        tp = len(gt_events & pr_events)
        fp = len(pr_events - gt_events)
        fn = len(gt_events - pr_events)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        rows.append((ct, precision, recall, f1))

    return pd.DataFrame(rows, columns=["cell_type", "precision", "recall", "f1"])

# --- Evaluation: CNV type per celltype + chromosome ---
def evaluate_cnv_with_window(adata, cnv_key="cnv", truth_key="simulated_cnvs", threshold_std=1.5, celltype_key="cell_type"):
    cnv = adata.obsm[f"X_{cnv_key}"]
    if issparse(cnv):
        cnv = cnv.toarray()

    std = np.std(cnv) + 1e-6
    chr_pos = adata.uns[cnv_key]["chr_pos"]
    chr_starts = list(chr_pos.values())
    chr_names = list(chr_pos.keys())

    rows = []
    for ct in adata.obs[celltype_key].unique():
        gt = adata.obs.loc[adata.obs[celltype_key] == ct, truth_key].explode()
        gt_events = [(x[0], x[1], x[2], x[3]) for x in gt]

        cell_mask = adata.obs[celltype_key] == ct
        for i, chrom in enumerate(chr_names):
            start_idx = chr_starts[i]
            end_idx = chr_starts[i+1] if i+1 < len(chr_starts) else cnv.shape[1]

            window = cnv[cell_mask, start_idx:end_idx]
            window_mean = window.mean(axis=0)
            calls = []
            for idx, val in enumerate(window_mean):
                if val > threshold_std * std:
                    calls.append((chrom, idx, idx+1, "gain"))
                elif val < -threshold_std * std:
                    calls.append((chrom, idx, idx+1, "loss"))

            pred = set(calls)
            gt_subset = {(c[0], c[1], c[2], c[3]) for c in gt_events if c[0] == chrom}

            tp = len(gt_subset & pred)
            fp = len(pred - gt_subset)
            fn = len(gt_subset - pred)

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            rows.append((ct, chrom, precision, recall, f1))

    return pd.DataFrame(rows, columns=["cell_type", "chromosome", "precision", "recall", "f1"])

