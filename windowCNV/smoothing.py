import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

# --- Helper: Natural chromosome sort ---
def _natural_sort(l: list[str]) -> list[str]:
    """Sort list with mixed numeric and string parts in natural order."""
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


# --- Compute convolution indices for smoothing ---
def get_convolution_indices(x: np.ndarray, n: int) -> np.ndarray:
    indices = []
    for i in range(x.shape[1] - n + 1):
        indices.append(np.arange(i, i + n))
    return np.array(indices)


# --- Pyramidially weighted running mean ---
def _running_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 50,
    step: int = 10,
    gene_list: list[str] = None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    if n < x.shape[1]:
        r = np.arange(1, n + 1)
        pyramid = np.minimum(r, r[::-1])
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        convolution_indices = get_convolution_indices(x, n)[np.arange(0, smoothed_x.shape[1], step)]
        convolved_gene_names = gene_list[convolution_indices]
        smoothed_x = smoothed_x[:, np.arange(0, smoothed_x.shape[1], step)]

        if calculate_gene_values:
            convolved_gene_values = _calculate_gene_averages(convolved_gene_names, smoothed_x)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values

    else:
        n = x.shape[1]
        pyramid = np.array([1] * n)
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        if calculate_gene_values:
            convolved_gene_values = pd.DataFrame(np.repeat(smoothed_x, len(gene_list), axis=1), columns=gene_list)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values


# --- Compute average value for each gene in the convolution ---
def _calculate_gene_averages(
    convolved_gene_names: np.ndarray,
    smoothed_x: np.ndarray,
) -> pd.DataFrame:
    gene_to_values = {}
    length = len(convolved_gene_names[0])
    flatten_list = list(convolved_gene_names.flatten())

    for sample, row in enumerate(smoothed_x):
        if sample not in gene_to_values:
            gene_to_values[sample] = {}
        for i, gene in enumerate(flatten_list):
            if gene not in gene_to_values[sample]:
                gene_to_values[sample][gene] = []
            gene_to_values[sample][gene].append(row[i // length])

    for sample in gene_to_values:
        for gene in gene_to_values[sample]:
            gene_to_values[sample][gene] = np.mean(gene_to_values[sample][gene])

    convolved_gene_values = pd.DataFrame(gene_to_values).T
    return convolved_gene_values


# --- Chromosome-wise running mean computation ---
def _running_mean_by_chromosome(
    expr,
    var,
    window_distance: float,
    step: int,
    calculate_gene_values: bool,
    min_genes_per_window: int = 5,
    smooth: bool = True
) -> tuple[dict, np.ndarray, list[int]]:
    chromosomes = _natural_sort([x for x in var["chromosome"].unique() if x.startswith("chr") and x != "chrM"])

    smoothed_chunks = []
    chr_pos = {}
    total_windows = 0
    n_genes_per_window = []

    for chrom in chromosomes:
        idxs = np.where(var['chromosome'] == chrom)[0]
        starts = var.iloc[idxs]['start'].values
        expr_chr = expr[:, idxs]

        sorted_order = np.argsort(starts)
        expr_chr = expr_chr[:, sorted_order]
        starts = starts[sorted_order]

        window_exprs = []
        win_start_idx = 0

        while win_start_idx < len(starts):
            win_start_pos = starts[win_start_idx]
            window_genes = []

            for j in range(win_start_idx, len(starts)):
                if starts[j] - win_start_pos <= window_distance:
                    window_genes.append(j)
                else:
                    break

            if len(window_genes) >= min_genes_per_window:
                if smooth:
                    window_expr = expr_chr[:, window_genes].mean(axis=1)
                else:
                    window_expr = expr_chr[:, window_genes].sum(axis=1)

                window_exprs.append(window_expr)
                n_genes_per_window.append(len(window_genes))

            win_start_idx = window_genes[-1] + 1 if len(window_genes) > 0 else win_start_idx + 1

        if len(window_exprs) > 0:
            chunk = np.vstack(window_exprs).T
            smoothed_chunks.append(chunk)
            chr_pos[chrom] = total_windows
            total_windows += chunk.shape[1]

    running_mean = np.hstack(smoothed_chunks)
    return chr_pos, running_mean, n_genes_per_window

