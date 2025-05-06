# plotting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scanpy as sc
from scipy.sparse import issparse
import re
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- Plotting: Heatmap of inferred CNV per cell ---
def plot_cna_heatmap(adata, chromosome, cell_type, layer="counts", celltype_key="cell_type"):
    """
    Plot heatmaps of CNA footprint for a given cell type and chromosome.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing gene expression data and CNA simulation.

    chromosome : str or int
        Chromosome to subset (e.g., '13' or 13).

    cell_type : str
        Cell type to subset (must match adata.obs[celltype_key] entries).

    layer : str, default='counts'
        Name of the layer to plot.

    celltype_key : str, default='cell_type'
        The name of the .obs column that contains cell type labels.

    Returns:
    --------
    None
    """
    import scanpy as sc
    chromosome = str(chromosome)
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"'{celltype_key}' not found in adata.obs.")

    adata_subset = adata[adata.obs[celltype_key] == cell_type].copy()
    adata_chr = adata_subset[:, adata_subset.var['chromosome'] == chromosome].copy()

    if adata_chr.n_obs == 0 or adata_chr.n_vars == 0:
        print(f"No data for cell type '{cell_type}' on chromosome {chromosome}.")
        return

    sc.pl.heatmap(
        adata_chr,
        adata_chr.var_names,
        groupby='simulated_cnvs',
        layer=layer,
        log=True,
        figsize=(20, 8),
        dendrogram=False,
        show_gene_labels=False,
        show=True
    )

    sc.pl.heatmap(
        adata_chr,
        adata_chr.var_names,
        groupby='simulated_cnvs',
        layer=layer,
        log=True,
        standard_scale='var',
        figsize=(20, 8),
        dendrogram=False,
        show_gene_labels=False,
        show=True
    )


# --- Plotting: Groundtruth + Inferred CNV map ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import issparse
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_groundtruth_and_inferred_cnv(
    adata,
    cell_type: str,
    celltype_key: str = "cell_type",
    cnv_truth_key: str = "simulated_cnvs",
    called_cnas_key: str = "called_cnas",
):
    # Define chromosomes and double them for gain/loss split
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chromosomes_split = [f"{chr}_gain" for chr in chromosomes] + [f"{chr}_loss" for chr in chromosomes]

    cells_idx = adata.obs.index[adata.obs[celltype_key] == cell_type].tolist()
    if not cells_idx:
        raise ValueError(f"No cells found for cell type '{cell_type}'.")

    n_cells = len(cells_idx)

    fig, axes = plt.subplots(1, 2, figsize=(24, 7))

    # ======== Groundtruth Frequency Matrix ========
    gain_freq = np.zeros(len(chromosomes))  # CN=4
    loss1_freq = np.zeros(len(chromosomes))  # CN=1
    loss0_freq = np.zeros(len(chromosomes))  # CN=0

    for cell_id in cells_idx:
        entry = adata.obs.loc[cell_id, cnv_truth_key]
        if pd.isna(entry) or entry.strip() == "":
            continue
        for event in str(entry).split(','):
            try:
                chrom_pos, cn_info = event.strip().split('(')
                chrom = chrom_pos.strip().split(':')[0]
                cn = int(cn_info.strip(' CN)').strip())
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"
                if chrom in chromosomes:
                    idx = chromosomes.index(chrom)
                    if cn == 0:
                        loss0_freq[idx] += 1
                    elif cn == 1:
                        loss1_freq[idx] += 1
                    elif cn == 4:
                        gain_freq[idx] += 1
            except Exception:
                continue

    gain_frac = gain_freq / n_cells
    loss1_frac = loss1_freq / n_cells
    loss0_frac = loss0_freq / n_cells

    # Construct matrix: rows = cells, cols = [chr1_gain, chr1_loss, ..., chrY_gain, chrY_loss]
    gt_matrix = np.zeros((n_cells, len(chromosomes_split)))
    for j, chr in enumerate(chromosomes):
        gain_col = j
        loss_col = j + len(chromosomes)
        gt_matrix[:, gain_col] = gain_frac[j]
        gt_matrix[:, loss_col] = loss0_frac[j] + 0.5 * loss1_frac[j]  # Darker for CN0

    cmap_gt = mcolors.ListedColormap(["white", "#3399FF", "#000066", "red"])
    norm_gt = mcolors.BoundaryNorm([0, 0.0001, 0.2, 0.5, 1.0], cmap_gt.N)

    im1 = axes[0].imshow(gt_matrix, aspect='auto', cmap=cmap_gt, norm=norm_gt)
    axes[0].set_xticks(np.arange(len(chromosomes_split)))
    axes[0].set_xticklabels(chromosomes_split, rotation=90, fontsize=6)
    axes[0].set_yticks([])
    axes[0].set_title(f"Groundtruth CNA Map (Gain vs Loss) - {cell_type}")
    plt.colorbar(im1, ax=axes[0], label="Fraction of Cells")

    # ======== Inferred CNA Frequency Matrix ========
    inferred_gain = np.zeros(len(chromosomes))
    inferred_loss = np.zeros(len(chromosomes))

    for cell_id in cells_idx:
        events = adata.obs.loc[cell_id, called_cnas_key]
        if events is None or isinstance(events, float) and np.isnan(events):
            continue
        for chrom, start, stop, ctype in events:
            if chrom in chromosomes:
                idx = chromosomes.index(chrom)
                if ctype == "gain":
                    inferred_gain[idx] += 1
                elif ctype == "loss":
                    inferred_loss[idx] += 1

    inferred_gain_frac = inferred_gain / n_cells
    inferred_loss_frac = inferred_loss / n_cells

    inf_matrix = np.zeros((n_cells, len(chromosomes_split)))
    for j, chr in enumerate(chromosomes):
        inf_matrix[:, j] = inferred_gain_frac[j]
        inf_matrix[:, j + len(chromosomes)] = inferred_loss_frac[j]

    im2 = axes[1].imshow(inf_matrix, aspect='auto', cmap=cmap_gt, norm=norm_gt)
    axes[1].set_xticks(np.arange(len(chromosomes_split)))
    axes[1].set_xticklabels(chromosomes_split, rotation=90, fontsize=6)
    axes[1].set_yticks([])
    axes[1].set_title(f"Inferred CNA Map (Gain vs Loss) - {cell_type}")
    plt.colorbar(im2, ax=axes[1], label="Fraction of Cells")

    plt.suptitle(f"Groundtruth vs Inferred CNA Maps - {cell_type}", fontsize=18)
    plt.tight_layout()
    plt.show()

# --- Plotting: Chromosome heatmap ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from anndata import AnnData
from scipy.sparse import issparse
from matplotlib.colors import TwoSlopeNorm

def plot_cnv_groundtruth_vs_inferred(
    adata: AnnData,
    cell_type: str,
    celltype_key: str = 'cell_type',
    cnv_truth_key: str | None = 'simulated_cnvs',
    cnv_inferred_key: str | None = 'cnv'
):
    """
    Plot groundtruth and/or inferred CNV maps for a given cell type.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cell_type : str
        Cell type to plot.
    celltype_key : str
        Key where cell types are stored in adata.obs.
    cnv_truth_key : str or None
        Key where groundtruth CNV annotations are stored in adata.obs. If None or False, skip this plot.
    cnv_inferred_key : str or None
        Key where inferred CNV matrix is stored in adata.obsm. If None or False, skip this plot.

    Returns
    -------
    None. Displays a matplotlib figure.
    """
    chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    cells = adata.obs[adata.obs[celltype_key] == cell_type]
    if cells.empty:
        raise ValueError(f"No cells found for cell type '{cell_type}'.")
    cells_idx = cells.index.tolist()

    # Determine number of panels
    do_truth = cnv_truth_key is not None and cnv_truth_key is not False
    do_infer = cnv_inferred_key is not None and cnv_inferred_key is not False
    n_panels = sum([do_truth, do_infer])
    if n_panels == 0:
        raise ValueError("At least one of cnv_truth_key or cnv_inferred_key must be set.")

    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 8))
    if n_panels == 1:
        axes = [axes]  # make iterable

    panel = 0

    # ----- Groundtruth Panel -----
    if do_truth:
        cnv_matrix_gt = np.full((len(cells_idx), len(chromosomes)), fill_value=-1)
        for idx, cell_id in enumerate(cells_idx):
            annotation = adata.obs.loc[cell_id, cnv_truth_key]
            if pd.isna(annotation) or annotation.strip() == '':
                continue
            for event in str(annotation).split(','):
                try:
                    chrom_pos, cn_info = event.strip().split('(')
                    chrom = chrom_pos.strip().split(':')[0]
                    cn = int(cn_info.strip(' CN)').strip())
                    chrom = f'chr{chrom}' if not chrom.startswith('chr') else chrom
                    if chrom in chromosomes:
                        chrom_idx = chromosomes.index(chrom)
                        if cn in [0, 1, 4]:
                            cnv_matrix_gt[idx, chrom_idx] = cn
                except Exception as e:
                    print(f"Warning parsing event '{event.strip()}': {e}")

        cmap_gt = mcolors.ListedColormap(["lightgray", "navy", "skyblue", "red"])
        bounds = [-1.5, -0.5, 0.5, 2.5, 5]
        norm_gt = mcolors.BoundaryNorm(bounds, cmap_gt.N)

        im = axes[panel].imshow(cnv_matrix_gt, aspect='auto', cmap=cmap_gt, norm=norm_gt, interpolation='nearest')
        axes[panel].set_xticks(np.arange(len(chromosomes)))
        axes[panel].set_xticklabels(chromosomes, rotation=90, fontsize=8)
        axes[panel].set_yticks([])
        axes[panel].set_title("Groundtruth CNV Map", fontsize=14)
        cbar = plt.colorbar(im, ax=axes[panel], boundaries=bounds, ticks=[-1, 0, 1, 4])
        cbar.ax.set_yticklabels(['No CNV', 'CN 0', 'CN 1', 'CN 4'])
        cbar.set_label('Groundtruth CNA')
        panel += 1

    # ----- Inferred Panel -----
    if do_infer:
        cnv_mtx = adata.obsm[f'X_{cnv_inferred_key}']
        if issparse(cnv_mtx):
            cnv_mtx = cnv_mtx.toarray()
        chr_pos_dict = dict(sorted(adata.uns[cnv_inferred_key]["chr_pos"].items(), key=lambda x: x[1]))
        chr_starts = list(chr_pos_dict.values())
        chr_slices = [slice(start, chr_starts[i + 1] if i + 1 < len(chr_starts) else cnv_mtx.shape[1])
                      for i, start in enumerate(chr_starts)]
        cnv_matrix_inf = np.zeros((len(cells_idx), len(chr_slices)))
        for i, sl in enumerate(chr_slices):
            cnv_matrix_inf[:, i] = cnv_mtx[adata.obs_names.get_indexer(cells_idx), sl].mean(axis=1)

        cmap_inf = plt.cm.coolwarm
        norm_inf = TwoSlopeNorm(0, vmin=np.nanmin(cnv_matrix_inf), vmax=np.nanmax(cnv_matrix_inf))
        im = axes[panel].imshow(cnv_matrix_inf, aspect='auto', cmap=cmap_inf, norm=norm_inf, interpolation='nearest')
        axes[panel].set_xticks(np.arange(len(chr_slices)))
        axes[panel].set_xticklabels(list(chr_pos_dict.keys()), rotation=90, fontsize=8)
        axes[panel].set_yticks([])
        axes[panel].set_title("Inferred CNV Map", fontsize=14)
        cbar = plt.colorbar(im, ax=axes[panel])
        cbar.set_label('Inferred CNV Score')

    plt.suptitle(f"CNV Map for '{cell_type}' ({len(cells_idx)} cells)", fontsize=16)
    plt.tight_layout()
    plt.show()

# --- Plotting: Inferred CNV matrix ---
def plot_inferred_cnv_map(
    adata,
    cell_type: str = None,
    chromosome: str = None,
    celltype_key: str = "cell_type",
    called_cnas_key: str = "called_cnas"
):
    """
    Plot Inferred CNV Map for specified cell type and chromosome.

    Parameters
    ----------
    adata : AnnData
        Annotated object with inferred CNAs in .obs[called_cnas_key].
    cell_type : str, optional
        If specified, plot only this cell type.
    chromosome : str, optional
        If specified, plot only this chromosome (e.g., 'chr7').
    celltype_key : str
        Column in .obs containing cell type labels.
    called_cnas_key : str
        Column in .obs containing CNA calls [(chr, start, stop, type)].

    Returns
    -------
    None. Displays a heatmap.
    """

    # Define all chromosomes in canonical order
    all_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chromosomes = [chromosome] if chromosome else all_chroms

    # Filter cells by cell type
    if cell_type:
        adata_subset = adata[adata.obs[celltype_key] == cell_type].copy()
    else:
        adata_subset = adata

    if adata_subset.shape[0] == 0:
        raise ValueError("No cells found for the specified filter.")

    # Create empty matrix: rows=cells, columns=chromosomes
    chrom_to_index = {chrom: i for i, chrom in enumerate(chromosomes)}
    cnv_matrix = np.zeros((adata_subset.shape[0], len(chromosomes)), dtype=int)

    # Fill matrix with gain (+1) / loss (-1) calls
    for i, cell_id in enumerate(adata_subset.obs_names):
        events = adata_subset.obs.at[cell_id, called_cnas_key]
        if not isinstance(events, list):
            continue
        for chrom, start, stop, event_type in events:
            if chrom in chrom_to_index:
                col = chrom_to_index[chrom]
                if event_type == "gain":
                    cnv_matrix[i, col] += 1
                elif event_type == "loss":
                    cnv_matrix[i, col] -= 1

    # Plot
    fig_width = max(6, 0.4 * len(chromosomes))
    plt.figure(figsize=(fig_width, 6))

    cmap = mcolors.ListedColormap(["blue", "white", "red"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    im = plt.imshow(cnv_matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    plt.xticks(np.arange(len(chromosomes)), chromosomes, rotation=90, fontsize=8)
    plt.yticks([])
    plt.title(f"Inferred CNV Map{' - ' + cell_type if cell_type else ''}", fontsize=14)

    cbar = plt.colorbar(im, boundaries=[-1.5, -0.5, 0.5, 1.5], ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Loss', 'Neutral', 'Gain'])
    cbar.set_label('Inferred CNA')

    plt.tight_layout()
    plt.show()

# --- Evaluation: Groundtruth vs inferred, original infercnv() ---
# Re-define the aligned CNV evaluation function
def evaluate_cnv_inference_aligned(
    adata,
    celltype_key='cell_type',
    cnv_truth_key='simulated_cnvs',
    cnv_inferred_key='cnv',
    threshold_std=1.0
):
    inferred = adata.obsm[f"X_{cnv_inferred_key}"]
    if issparse(inferred):
        inferred = inferred.toarray()
    inferred /= (np.std(inferred) + 1e-6)

    chr_pos_dict = dict(sorted(adata.uns[cnv_inferred_key]["chr_pos"].items(), key=lambda x: x[1]))
    chr_names = list(chr_pos_dict.keys())
    chr_starts = list(chr_pos_dict.values())
    chr_slices = {}

    for i in range(len(chr_names)):
        chrom = chr_names[i]
        start = chr_starts[i]
        end = chr_starts[i + 1] if i + 1 < len(chr_names) else inferred.shape[1]
        chr_slices[chrom] = slice(start, end)

    pattern = r"([\w]+):\d+-\d+ \(CN (\d)\)"
    event_map = {}

    for idx, row in adata.obs.iterrows():
        ct = row[celltype_key]
        val = row.get(cnv_truth_key, "")
        if not isinstance(val, str) or val.strip() == "":
            continue
        matches = re.findall(pattern, val)
        for chrom, cn in matches:
            chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
            cn = int(cn)
            if cn == 2:
                continue
            label = "gain" if cn > 2 else "loss"
            key = (ct, chrom, cn, label)
            if key not in event_map:
                event_map[key] = []
            event_map[key].append(idx)

    results = []
    for (celltype, chrom, cn, label), pos_cells in event_map.items():
        event_name = f"{celltype}_{chrom}_CN{cn}_{label}"

        if chrom not in chr_slices:
            results.append({
                "Event": event_name,
                "TP": None, "FP": None, "FN": None,
                "Precision": None, "Recall": None, "F1": None, "Accuracy": None
            })
            continue

        slice_idx = chr_slices[chrom]
        direction = 1 if label == "gain" else -1

        all_cells = adata.obs[adata.obs[celltype_key] == celltype].index
        pos_mask = np.isin(all_cells, pos_cells)
        gt = np.zeros(len(all_cells), dtype=int)
        gt[pos_mask] = 1

        cell_idx = adata.obs_names.get_indexer(all_cells)
        scores = inferred[cell_idx, slice_idx].mean(axis=1)
        pred_raw = np.zeros_like(scores)
        pred_raw[scores > threshold_std] = 1
        pred_raw[scores < -threshold_std] = -1
        pred = (pred_raw == direction).astype(int)

        TP = np.sum((gt == 1) & (pred == 1))
        FP = np.sum((gt == 0) & (pred == 1))
        FN = np.sum((gt == 1) & (pred == 0))

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        acc = TP / (TP + FP + FN + 1e-6)

        results.append({
            "Event": event_name,
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": acc
        })

    df = pd.DataFrame(results).set_index("Event")

    plt.figure(figsize=(10, 1 * len(df)))
    sns.heatmap(
        df[["Precision", "Recall", "F1", "Accuracy"]],
        annot=True,
        fmt=".4f",
        cmap=sns.light_palette("red", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='black'
    )
    plt.title("CNV Inference Metrics per Event (Gene-based)")
    plt.tight_layout()
    plt.show()

    return df

# --- Evaluation: Groundtruth vs inferred, with our window ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.sparse import issparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define the modified function based on direct threshold comparison
def evaluate_cnv_with_window(
    adata,
    celltype_key='cell_type',
    cnv_truth_key='simulated_cnvs',
    cnv_inferred_key='cnv',
    gain_percentile=90,
    loss_percentile=10
):
    inferred = adata.obsm[f"X_{cnv_inferred_key}"]
    if issparse(inferred):
        inferred = inferred.toarray()

    # --- Percentile-based thresholds ---
    flat_vals = inferred.flatten()
    pos_vals = flat_vals[flat_vals > 0]
    neg_vals = flat_vals[flat_vals < 0]
    
    if len(pos_vals) > 0:
        gain_threshold = np.percentile(pos_vals, 100 - gain_percentile)
    else:
        gain_threshold = np.inf  # Effectively disables gain detection
    
    if len(neg_vals) > 0:
        loss_threshold = np.percentile(neg_vals, loss_percentile)
    else:
        loss_threshold = -np.inf  # Effectively disables loss detection
    
    print(f"[INFO] Gain threshold: > {gain_threshold:.4f}")
    print(f"[INFO] Loss threshold: < {loss_threshold:.4f}")

    # --- Existing unchanged logic ---
    chr_pos_dict = dict(sorted(adata.uns[cnv_inferred_key]["chr_pos"].items(), key=lambda x: x[1]))
    window_index_map = []
    chr_keys = list(chr_pos_dict.keys())
    chr_vals = list(chr_pos_dict.values())
    for i, chrom in enumerate(chr_keys):
        start = chr_vals[i]
        end = chr_vals[i + 1] if i + 1 < len(chr_vals) else inferred.shape[1]
        for j in range(start, end):
            window_index_map.append((chrom, j))
    window_df = pd.DataFrame(window_index_map, columns=["chromosome", "matrix_idx"])

    pattern = r"(\w+):(\d+)-(\d+)\s+\(CN\s+(\d)\)"
    grouped_results = {}
    unmatched_fp_keys = set()
    excluded_event_keys = set()
    printed_exclusions = set()

    gt_event_set = set()
    for idx, row in adata.obs.iterrows():
        ct = row[celltype_key]
        annots = row.get(cnv_truth_key, "")
        row_idx = adata.obs_names.get_loc(idx)
        if isinstance(annots, str) and annots.strip() != "":
            matches = re.findall(pattern, annots)
            for chrom, start, end, cn in matches:
                chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                cn = int(cn)
                if cn == 2:
                    continue
                gt = "gain" if cn > 2 else "loss"
                gt_event_set.add((ct, chrom, cn, gt))

    for idx, row in adata.obs.iterrows():
        ct = row[celltype_key]
        annots = row.get(cnv_truth_key, "")
        row_idx = adata.obs_names.get_loc(idx)

        if isinstance(annots, str) and annots.strip() != "":
            matches = re.findall(pattern, annots)
            for chrom, start, end, cn in matches:
                chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                cn = int(cn)
                if cn == 2:
                    continue
                gt = "gain" if cn > 2 else "loss"
                win_idxs = window_df[window_df["chromosome"] == chrom]["matrix_idx"].values
                key = (ct, chrom, cn, gt)
                if len(win_idxs) == 0:
                    if key not in printed_exclusions:
                        print(f"Excluded GT CNV event due to missing chromosome: {key}")
                        printed_exclusions.add(key)
                    excluded_event_keys.add(key)
                    continue
                win_vals = inferred[row_idx, win_idxs]
                max_val = win_vals.max()
                min_val = win_vals.min()
                pred = "gain" if max_val > gain_threshold else (
                    "loss" if min_val < loss_threshold else "no_change"
                )
                pred_label = int(pred == gt)
                if key not in grouped_results:
                    grouped_results[key] = {"true": [], "pred": []}
                grouped_results[key]["true"].append(1)
                grouped_results[key]["pred"].append(pred_label)

        for chrom in window_df["chromosome"].unique():
            win_idxs = window_df[window_df["chromosome"] == chrom]["matrix_idx"].values
            if len(win_idxs) == 0:
                continue
            win_vals = inferred[row_idx, win_idxs]
            max_val = win_vals.max()
            min_val = win_vals.min()
            pred = "gain" if max_val > gain_threshold else (
                "loss" if min_val < loss_threshold else "no_change"
            )
            if pred == "no_change":
                continue
            for ct_gt, chr_gt, cn_gt, gt_label in gt_event_set:
                if ct_gt == ct and gt_label == pred and chr_gt == chrom:
                    key = (ct, chr_gt, cn_gt, gt_label)
                    if key not in excluded_event_keys:
                        if key not in grouped_results:
                            grouped_results[key] = {"true": [], "pred": []}
                        grouped_results[key]["true"].append(0)
                        grouped_results[key]["pred"].append(1)

    # Build final dataframe
    metrics_rows = []
    all_keys = gt_event_set.union(excluded_event_keys)
    for key in sorted(all_keys):
        ct, chrom, cn, gt = key
        if key in excluded_event_keys:
            metrics_rows.append({
                "cell_type": ct, "chromosome": chrom, "CN": cn, "groundtruth": gt,
                "TP": None, "FP": None, "FN": None,
                "precision": None, "recall": None, "f1": None,
                "accuracy": None, "n_events": 0
            })
        else:
            y_true = grouped_results[key]["true"]
            y_pred = grouped_results[key]["pred"]
            TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            metrics_rows.append({
                "cell_type": ct, "chromosome": chrom, "CN": cn, "groundtruth": gt,
                "TP": TP, "FP": FP, "FN": FN,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "n_events": len(y_true)
            })

    df = pd.DataFrame(metrics_rows)
    df["CN"] = df["CN"].astype(str)

    heatmap_df = df.set_index(["cell_type", "chromosome", "CN", "groundtruth"])[
        ["precision", "recall", "f1", "accuracy"]
    ]
    heatmap_df.index = heatmap_df.index.map(lambda x: f"{x[0]}_{x[1]}_CN{x[2]}_{x[3]}")
    heatmap_df.index.name = "CNV Event"

    plt.figure(figsize=(10, 1 * len(heatmap_df)))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".4f",
        cmap=sns.light_palette("red", as_cmap=True),
        linewidths=0.5, linecolor="black",
        mask=heatmap_df.isnull(), cbar_kws={'label': 'Score'},
        vmin=0.0, vmax=1.0
    )
    plt.title("CNV Inference Metrics per Event (Window-based)")
    plt.tight_layout()
    plt.show()

    n_predicted = sum([sum(x['pred']) for x in grouped_results.values()])
    print(f"[INFO] Total predicted CNVs: {n_predicted}")

    return df, excluded_event_keys
