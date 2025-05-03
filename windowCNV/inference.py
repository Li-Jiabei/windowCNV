from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse
from typing import Sequence

from anndata import AnnData
from tqdm.auto import tqdm
from infercnvpy._util import _ensure_array
from .smoothing import _running_mean_by_chromosome
from joblib import Parallel, delayed
from scipy.sparse import issparse

# --- CNV Inference Core ---

def infercnv(
    adata: AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    reference: np.ndarray | None = None,
    normalization_mode: str = "reference",
    lfc_clip: float = 3,
    window_distance: float = 5e6,
    min_genes_per_window: int = 5,
    smooth: bool = True,
    dynamic_threshold: float | None = 1.5,
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    chunksize: int = 5000,
    n_jobs: int | None = None,
    inplace: bool = True,
    layer: str | None = None,
    key_added: str = "cnv",
    calculate_gene_values: bool = False,
    two_step_refinement: bool = False,
    low_variance_quantile: float = 0.3,
) -> None | tuple[dict, scipy.sparse.csr_matrix, np.ndarray | None]:
    """Infer Copy Number Variation (CNV) by averaging gene expression over genomic regions."""

    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")

    if {"chromosome", "start", "end"} - set(adata.var.columns):
        raise ValueError("`chromosome`, `start`, and `end` must be in adata.var.")

    # Auto-add "chr" prefix
    if adata.var["chromosome"].dtype == object:
        unique_chroms = adata.var["chromosome"].dropna().unique()
        if all(not str(chrom).startswith("chr") for chrom in unique_chroms):
            adata.var["chromosome"] = adata.var["chromosome"].apply(
                lambda x: f"chr{x}" if pd.notna(x) else x
            )

    var_mask = adata.var["chromosome"].isnull()
    if np.sum(var_mask):
        print(f"Skipped {np.sum(var_mask)} genes without genomic positions.")

    if exclude_chromosomes is not None:
        var_mask |= adata.var["chromosome"].isin(exclude_chromosomes)

    tmp_adata = adata[:, ~var_mask]
    reference = _get_reference(tmp_adata, reference_key, reference_cat, reference)
    expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]
    expr = expr.tocsr() if scipy.sparse.issparse(expr) else np.asarray(expr)
    var = tmp_adata.var.loc[:, ["chromosome", "start", "end"]]

    if two_step_refinement:
        print("Running two-step refinement...")
        expr_z = (expr - np.mean(expr, axis=0)) / (np.std(expr, axis=0) + 1e-6)
        chr_pos, rough_cnv, _ = _infercnv_chunk(expr_z, var, reference, lfc_clip,
                                                window_distance, min_genes_per_window,
                                                True, dynamic_threshold)
        rough_cnv = rough_cnv.toarray()
        cell_vars = np.var(rough_cnv, axis=1)
        threshold = np.quantile(cell_vars, low_variance_quantile)
        stable_mask = cell_vars <= threshold
        if not np.any(stable_mask):
            raise ValueError("No low-variance cells selected. Raise quantile.")
        reference = np.mean(expr[stable_mask, :], axis=0, keepdims=True)
        print(f"Selected {np.sum(stable_mask)} cells for refined reference.")
    elif normalization_mode == "reference":
        reference = _get_reference(tmp_adata, reference_key, reference_cat, reference)
    elif normalization_mode == "auto":
        reference = np.mean(expr, axis=0, keepdims=True)
    else:
        raise ValueError("normalization_mode must be 'reference' or 'auto'")

    chunks = []
    chr_pos_final = None

    for i in tqdm(range(0, adata.shape[0], chunksize), desc="Running inferCNV chunks"):
        chunk_expr = expr[i: i + chunksize, :]
        chr_pos, chunk, _ = _infercnv_chunk(
            chunk_expr, var, reference, lfc_clip, window_distance,
            min_genes_per_window, smooth, dynamic_threshold
        )
        if chr_pos_final is None:
            chr_pos_final = chr_pos
        chunks.append(chunk)

    res = scipy.sparse.vstack(chunks)

    if inplace:
        adata.obsm[f"X_{key_added}"] = res
        adata.uns[key_added] = {"chr_pos": chr_pos_final}
    else:
        return chr_pos_final, res, None

# --- CNV Chunk Logic ---

def _infercnv_chunk(
    tmp_x,
    var,
    reference,
    lfc_cap,
    window_distance,
    min_genes_per_window,
    smooth,
    dynamic_threshold,
    calculate_gene_values=False
):
    if reference.shape[0] == 1:
        x_centered = tmp_x - reference[0, :]
    else:
        ref_min = np.min(reference, axis=0)
        ref_max = np.max(reference, axis=0)
        x_centered = np.zeros(tmp_x.shape, dtype=tmp_x.dtype)
        above_max = tmp_x > ref_max
        below_min = tmp_x < ref_min
        x_centered[above_max] = _ensure_array(tmp_x - ref_max)[above_max]
        x_centered[below_min] = _ensure_array(tmp_x - ref_min)[below_min]

    x_clipped = np.clip(_ensure_array(x_centered), -lfc_cap, lfc_cap)
    chr_pos, running_mean, _ = _running_mean_by_chromosome(
        x_clipped, var,
        window_distance=window_distance,
        step=1,
        calculate_gene_values=False,
        min_genes_per_window=min_genes_per_window,
        smooth=smooth
    )
    x_res = running_mean - np.median(running_mean, axis=1)[:, np.newaxis]
    if dynamic_threshold is not None:
        noise_thres = dynamic_threshold * np.std(x_res)
        x_res[np.abs(x_res) < noise_thres] = 0
    return chr_pos, scipy.sparse.csr_matrix(x_res), None

# --- CNA Assignment ---

def assign_cnas_to_cells_parallel(
    adata: AnnData,
    cnv_key: str = "cnv",
    threshold_std: float = 1.5,
    output_key: str = "called_cnas",
    n_jobs: int = -1,
):
    """Assign CNAs to cells using CNV signal, parallelized with joblib."""
    cnv = adata.obsm[f"X_{cnv_key}"]
    cnv = cnv.toarray() if issparse(cnv) else cnv
    cnv_std = cnv / (np.std(cnv) + 1e-6)
    chr_pos = adata.uns[cnv_key]["chr_pos"]
    chr_list = list(chr_pos.keys())
    chr_starts = list(chr_pos.values())

    chr_slices = {
        chrom: (chr_starts[i], chr_starts[i + 1] if i + 1 < len(chr_starts) else cnv.shape[1])
        for i, chrom in enumerate(chr_list)
    }

    gene_windows = adata.var.loc[:, ["chromosome", "start", "end"]].reset_index(drop=True)
    if len(gene_windows) < cnv.shape[1]:
        gene_windows = pd.concat(
            [gene_windows] * (cnv.shape[1] // len(gene_windows) + 1),
            ignore_index=True
        )
    gene_windows = gene_windows.iloc[:cnv.shape[1]]

    def process_cell(cell_idx):
        events = []
        for chrom, (start_idx, end_idx) in chr_slices.items():
            for i in range(start_idx, end_idx):
                score = cnv_std[cell_idx, i]
                if abs(score) >= threshold_std:
                    event_type = "gain" if score > 0 else "loss"
                    start = gene_windows.iloc[i]["start"]
                    stop = gene_windows.iloc[i]["end"]
                    events.append((chrom, int(start), int(stop), event_type))
        return events

    print(f"Assigning CNAs using {n_jobs if n_jobs > 0 else 'all'} cores...")
    from tqdm.auto import tqdm
    cell_cnas = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(i) for i in tqdm(range(cnv.shape[0]), desc="Parallel CNA assignment")
    )
    adata.obs[output_key] = cell_cnas

