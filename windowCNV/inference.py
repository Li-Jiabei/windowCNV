from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse
import re
import logging
from typing import Sequence

from anndata import AnnData
from tqdm.auto import tqdm
from infercnvpy._util import _ensure_array
from .smoothing import _running_mean_by_chromosome
from joblib import Parallel, delayed
from scipy.sparse import issparse
from .preprocess import _get_reference


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
    calculate_gene_values: bool = False
)-> None | tuple[dict, scipy.sparse.csr_matrix, np.ndarray | None]:
    """Infer Copy Number Variation (CNV) by averaging gene expression over genomic regions.

    This method is heavily inspired by `infercnv <https://github.com/broadinstitute/inferCNV/>`_
    but more computationally efficient. The method is described in more detail
    in on the :ref:`infercnv-method` page.

    There, you can also find instructions on how to :ref:`prepare input data <input-data>`.

    Parameters
    ----------
    adata
        annotated data matrix
    reference_key
        Column name in adata.obs that contains tumor/normal annotations.
        If this is set to None, the average of all cells is used as reference.
    reference_cat
        One or multiple values in `adata.obs[reference_key]` that annotate
        normal cells.
    reference
        Directly supply an array of average normal gene expression. Overrides
        `reference_key` and `reference_cat`.
    normalization_mode
        Normalization mode. Options:
        - `"reference"`: Normalize using a supplied or inferred reference profile.
        - `"auto"`: Automatically calculate a reference as the mean expression across all cells.
    lfc_clip
        Clip log fold changes at this value
    window distance
    min_genes_per_window
    smooth
    dynamic_threshold
        Values `< dynamic threshold * STDDEV` will be set to 0, where STDDEV is
        the stadard deviation of the smoothed gene expression. Set to `None` to disable
        this step.
    exclude_chromosomes
        List of chromosomes to exclude. The default is to exclude genosomes.
    chunksize
        Process dataset in chunks of cells. This allows to run infercnv on
        datasets with many cells, where the dense matrix would not fit into memory.
    n_jobs
        Number of jobs for parallel processing. Default: use all cores.
        Data will be submitted to workers in chunks, see `chunksize`.
    inplace
        If True, save the results in adata.obsm, otherwise return the CNV matrix.
    layer
        Layer from adata to use. If `None`, use `X`.
    key_added
        Key under which the cnv matrix will be stored in adata if `inplace=True`.
        Will store the matrix in `adata.obsm["X_{key_added}"] and additional information
        in `adata.uns[key_added]`.
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
        As many genes will be included in each segment the resultant per gene value will be an average of the genes included in the segment.
        Additionally not all genes will be included in the per gene CNV, due to the window size and step size not always being a multiple of
        the number of genes. Any genes not included in the per gene CNV will be filled with NaN.
        Note this will significantly increase the memory and computation time, it is recommended to decrease the chunksize to ~100 if this is set to True.

    Returns
    -------
    Depending on inplace, either return the smoothed and denoised gene expression
    matrix sorted by genomic position, or add it to adata.
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns) != set():
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and `end` columns in `adata.var`. "
        )
    
    # Ensure chromosomes start with 'chr'
    # Standardize chromosome format and filter to standard chromosomes
    adata.var['chromosome'] = adata.var['chromosome'].apply(
        lambda x: f'chr{x}' if pd.notna(x) and not str(x).startswith('chr') else x
    )
    standard_chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    chrom_mask = adata.var['chromosome'].isin(standard_chromosomes)
    
    if not inplace:
        adata = adata[:, chrom_mask].copy()
    else:
        adata._inplace_subset_var(chrom_mask)

    var_mask = adata.var["chromosome"].isnull()
    if np.sum(var_mask):
        logging.warning(f"Skipped {np.sum(var_mask)} genes because they don't have a genomic position annotated. ")  # type: ignore
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var["chromosome"].isin(exclude_chromosomes)

    # === Determine expression matrix and var ===
    tmp_adata = adata[:, ~var_mask]

    # call reference on filtered adata, so shapes match
    reference = _get_reference(tmp_adata, reference_key, reference_cat, reference)

    expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]

    if scipy.sparse.issparse(expr):
        expr = expr.tocsr()
    else:
        expr = np.asarray(expr)

    var = tmp_adata.var.loc[:, ["chromosome", "start", "end"]]

    # === Reference selection and normalization ===
    if normalization_mode == "reference":
        reference = _get_reference(tmp_adata, reference_key, reference_cat, reference)
    elif normalization_mode == "auto":
        logging.info("Using in-built average reference from all cells.")
        reference = np.mean(expr, axis=0, keepdims=True)
    else:
        raise ValueError("normalization_mode must be 'reference' or 'auto'")

    chunks = []
    chr_pos_list = []
    convolved_dfs = []
    chr_pos_final = None

    for i in tqdm(range(0, adata.shape[0], chunksize), desc="Running inferCNV chunks"):
        chunk_expr = expr[i : i + chunksize, :]

        chr_pos, chunk, _ = _infercnv_chunk(
            chunk_expr,
            var,
            reference,
            lfc_clip,
            window_distance,
            min_genes_per_window,
            smooth,
            dynamic_threshold
        )
    
        if chr_pos_final is None:
            chr_pos_final = chr_pos  # use the first one

        chunks.append(chunk)
        convolved_dfs.append(pd.DataFrame())  # Placeholder if needed

    res = scipy.sparse.vstack(chunks)
    chr_pos = chr_pos_final

    if calculate_gene_values:
        per_gene_df = pd.concat(convolved_dfs, axis=0)
        # Ensure the DataFrame has the correct row index
        per_gene_df.index = adata.obs.index
        # Ensure the per gene CNV matches the adata var (genes) index, any genes
        # that are not included in the CNV will be filled with NaN
        per_gene_df = per_gene_df.reindex(columns=adata.var_names, fill_value=np.nan)
        # This needs to be a numpy array as colnames are too large to save in anndata
        per_gene_mtx = per_gene_df.values
    else:
        per_gene_mtx = None

    if inplace:

        adata.obsm[f"X_{key_added}"] = res
        adata.uns[key_added] = {"chr_pos": chr_pos}

        if calculate_gene_values:
            adata.layers[f"gene_values_{key_added}"] = per_gene_mtx

    else:
        return chr_pos, res, per_gene_mtx

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
    """
    The actual infercnv work is happening here.

    Process chunks of serveral thousand genes independently since this
    leads to (temporary) densification of the matrix.

    Parameters see `infercnv`.
    """
    """Chunk-based adaptive CNV inference: log fold-change, smoothing, denoising."""

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

    x_centered = _ensure_array(x_centered)
    x_clipped = np.clip(x_centered, -lfc_cap, lfc_cap)

    chr_pos, running_mean, _ = _running_mean_by_chromosome(
        x_clipped,
        var,
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

    x_res = scipy.sparse.csr_matrix(x_res)

    return chr_pos, x_res, None

# --- CNA Assignment ---
from joblib import Parallel, delayed
from scipy.sparse import issparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def assign_cnas_to_cells_parallel(
    adata,
    cnv_key: str = "cnv",
    gain_percentile: float = 30,
    loss_percentile: float = 30,
    output_key: str = "called_cnas",
    n_jobs: int = -1,
):
    """
    Assign CNAs based on percentile thresholds (most extreme 30% by default).
    Stores list of (chr, start, end, type) in adata.obs[output_key].

    Parameters
    ----------
    adata : AnnData
        AnnData object with per-cell CNV scores in `.obsm['X_<cnv_key>']`.
    cnv_key : str
        Key for CNV signal matrix (default: "cnv" → adata.obsm["X_cnv"]).
    gain_percentile : float
        Percentile of most positive values to consider as gain (e.g., 30 → top 30% of positives).
    loss_percentile : float
        Percentile of most negative values to consider as loss (e.g., 30 → bottom 30% of negatives).
    output_key : str
        Key to store result in `adata.obs`.
    n_jobs : int
        Number of parallel workers (default: all).
    """

    # Load CNV signal
    cnv = adata.obsm[f"X_{cnv_key}"]
    if issparse(cnv):
        cnv = cnv.toarray()

    flat_vals = cnv.flatten()
    pos_vals = flat_vals[flat_vals > 0]
    neg_vals = flat_vals[flat_vals < 0]

    # Compute thresholds from specified percentiles of pos/neg only
    gain_threshold = np.percentile(pos_vals, 100 - gain_percentile) if len(pos_vals) else np.inf
    loss_threshold = np.percentile(neg_vals, loss_percentile) if len(neg_vals) else -np.inf

    print(f"[INFO] Gain threshold: > {gain_threshold:.4f}")
    print(f"[INFO] Loss threshold: < {loss_threshold:.4f}")

    # Position indices by chromosome
    chr_pos = adata.uns[cnv_key]["chr_pos"]
    chr_list = list(chr_pos.keys())
    chr_starts = list(chr_pos.values())
    chr_slices = {
        chrom: (chr_starts[i], chr_starts[i+1] if i+1 < len(chr_starts) else cnv.shape[1])
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
                score = cnv[cell_idx, i]
                if score > gain_threshold:
                    event_type = "gain"
                elif score < loss_threshold:
                    event_type = "loss"
                else:
                    continue
                start = gene_windows.iloc[i]["start"]
                stop = gene_windows.iloc[i]["end"]
                events.append((chrom, int(start), int(stop), event_type))
        return events

    print(f"Assigning CNAs using {n_jobs if n_jobs > 0 else 'all'} cores...")

    cell_cnas = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(i) for i in tqdm(range(cnv.shape[0]), desc="Parallel CNA assignment")
    )

    adata.obs[output_key] = cell_cnas

