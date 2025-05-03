import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

def _natural_sort(l: Sequence):
    """Natural sort without third party libraries.

    Adapted from: https://stackoverflow.com/a/4836734/2340703
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def _running_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 50,
    step: int = 10,
    gene_list: list = None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Compute a pyramidially weighted running mean.

    Densifies the matrix. Use `step` and `chunksize` to save memory.

    Parameters
    ----------
    x
        matrix to work on
    n
        Length of the running window
    step
        only compute running windows every `step` columns, e.g. if step is 10
        0:99, 10:109, 20:119 etc. Saves memory.
    gene_list
        List of gene names to be used in the convolution
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
    """
    if n < x.shape[1]:  # regular convolution: the filter is smaller than the #genes
        r = np.arange(1, n + 1)
        pyramid = np.minimum(r, r[::-1])
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        ## get the indices of the genes used in the convolution
        convolution_indices = get_convolution_indices(x, n)[np.arange(0, smoothed_x.shape[1], step)]
        ## Pull out the genes used in the convolution
        convolved_gene_names = gene_list[convolution_indices]
        smoothed_x = smoothed_x[:, np.arange(0, smoothed_x.shape[1], step)]

        if calculate_gene_values:
            convolved_gene_values = _calculate_gene_averages(convolved_gene_names, smoothed_x)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values

    else:  # If there is less genes than the window size, set the window size to the number of genes and perform a single convolution
        n = x.shape[1]  # set the filter size to the number of genes
        r = np.arange(1, n + 1)
        ## As we are only doing one convolution the values should be equal
        pyramid = np.array([1] * n)
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        if calculate_gene_values:
            ## As all genes are used the convolution the values are identical for all genes
            convolved_gene_values = pd.DataFrame(np.repeat(smoothed_x, len(gene_list), axis=1), columns=gene_list)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values


def _calculate_gene_averages(
    convolved_gene_names: np.ndarray,
    smoothed_x: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate the average value of each gene in the convolution

    Parameters
    ----------
    convolved_gene_names
        A numpy array with the gene names used in the convolution
    smoothed_x
        A numpy array with the smoothed gene expression values

    Returns
    -------
    convolved_gene_values
        A DataFrame with the average value of each gene in the convolution
    """
    ## create a dictionary to store the gene values per sample
    gene_to_values = {}
    # Calculate the number of genes in each convolution, will be same as the window size default=100
    length = len(convolved_gene_names[0])
    # Convert the flattened convolved gene names to a list
    flatten_list = list(convolved_gene_names.flatten())

    # For each sample in smoothed_x find the value for each gene and store it in a dictionary
    for sample, row in enumerate(smoothed_x):
        # Create sample level in the dictionary
        if sample not in gene_to_values:
            gene_to_values[sample] = {}
        # For each gene in the flattened gene list find the value and store it in the dictionary
        for i, gene in enumerate(flatten_list):
            if gene not in gene_to_values[sample]:
                gene_to_values[sample][gene] = []
            # As the gene list has been flattend we can use the floor division of the index
            # to get the correct position of the gene to get the value and store it in the dictionary
            gene_to_values[sample][gene].append(row[i // length])

    for sample in gene_to_values:
        for gene in gene_to_values[sample]:
            gene_to_values[sample][gene] = np.mean(gene_to_values[sample][gene])

    convolved_gene_values = pd.DataFrame(gene_to_values).T
    return convolved_gene_values


def get_convolution_indices(x, n):
    indices = []
    for i in range(x.shape[1] - n + 1):
        indices.append(np.arange(i, i + n))
    return np.array(indices)


def _running_mean_by_chromosome(
    expr,
    var,
    window_distance,  # genomic bp
    step,  # unused, kept for compatibility
    calculate_gene_values,
    min_genes_per_window=5,
    smooth=True
) -> tuple[dict, np.ndarray, list[int]]:
    """
    Compute the running mean for each chromosome independently. Stack the resulting arrays ordered by chromosome.

    Returns
    -------
    chr_start_pos
        A Dictionary mapping each chromosome to the index of running_mean where
        this chromosome begins.
    running_mean
        A numpy array with the smoothed gene expression, ordered by chromosome
        and genomic position
    n_genes_per_window : list[int]
        Number of genes used in each window.
    """

    chromosomes = _natural_sort([x for x in var["chromosome"].unique() if x.startswith("chr") and x != "chrM"])

    smoothed_chunks = []
    chr_pos = {}
    total_windows = 0
    n_genes_per_window = []  # store number of genes per window

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
            # only accept windows with enough genes
            if len(window_genes) >= min_genes_per_window:
                if smooth:
                    window_expr = expr_chr[:, window_genes].mean(axis=1)
                else:
                    window_expr = expr_chr[:, window_genes].sum(axis=1)

                window_exprs.append(window_expr)
                n_genes_per_window.append(len(window_genes))  # record number of genes used

            win_start_idx = window_genes[-1] + 1 if len(window_genes) > 0 else win_start_idx + 1

        if len(window_exprs) > 0:
            chunk = np.vstack(window_exprs).T
            smoothed_chunks.append(chunk)
            chr_pos[chrom] = total_windows
            total_windows += chunk.shape[1]

    running_mean = np.hstack(smoothed_chunks)

    return chr_pos, running_mean, n_genes_per_window  # return genes per window too
