import numpy as np
import pandas as pd
import scipy.sparse
from typing import Sequence
from anndata import AnnData
import logging


def find_reference_candidates(adata: AnnData, reference_key: str = "cell_type", top_n: int = 5):
    """
    Find reference cell types based on lowest average variance across all genes/features.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression (or CNV scores) in `.X` and annotations in `.obs`.
    reference_key : str
        Column in `.obs` used to group cells (typically 'cell_type').
    top_n : int
        Number of lowest-variance cell types to return.

    Returns
    -------
    List[str]
        Names of the top_n lowest-variance categories.
    """

    # Get unique categories (e.g., all cell types)
    categories = adata.obs[reference_key].unique()

    results = []

    # Iterate over each category
    for cat in categories:
        # Select cells belonging to the current category
        subset = adata[adata.obs[reference_key] == cat].X

        # Convert to dense if sparse
        if scipy.sparse.issparse(subset):
            subset = subset.toarray()

        # Calculate average variance across genes/features
        avg_var = np.var(subset, axis=0).mean()

        # Store result: (category name, average variance)
        results.append((cat, avg_var))

    # Sort categories by ascending average variance
    results.sort(key=lambda x: x[1])

    # Print out the top N
    print(f"\nTop {top_n} lowest-variance cell types:\n")
    for cat, var in results[:top_n]:
        print(f"{cat}: avg variance = {var:.4f}")

    # Return list of top N category names
    return [cat for cat, _ in results[:top_n]]


def _get_reference(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: None | str | Sequence[str],
    reference: np.ndarray | None,
) -> np.ndarray:
    """Parameter validation extraction of reference gene expression.

    If multiple reference categories are given, compute the mean per
    category.

    Returns a 2D array with reference categories in rows, cells in columns.
    If there's just one category, it's still a 2D array.
    """
    if reference is None:
        if reference_key is None or reference_cat is None:
            logging.warning(
                "Using mean of all cells as reference. For better results, "
                "provide either `reference`, or both `reference_key` and `reference_cat`. "
            )  # type: ignore
            reference = np.mean(adata.X, axis=0)

        else:
            obs_col = adata.obs[reference_key]
            if isinstance(reference_cat, str):
                reference_cat = [reference_cat]
            reference_cat = np.array(reference_cat)
            reference_cat_in_obs = np.isin(reference_cat, obs_col)
            if not np.all(reference_cat_in_obs):
                raise ValueError(
                    "The following reference categories were not found in "
                    "adata.obs[reference_key]: "
                    f"{reference_cat[~reference_cat_in_obs]}"
                )

            reference = np.vstack([np.mean(adata.X[obs_col.values == cat, :], axis=0) for cat in reference_cat])

    if reference.ndim == 1:
        reference = reference[np.newaxis, :]

    if reference.shape[1] != adata.shape[1]:
        raise ValueError("Reference must match the number of genes in AnnData. ")

    return reference

