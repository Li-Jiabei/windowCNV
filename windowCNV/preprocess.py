import numpy as np
import pandas as pd
import scipy.sparse
from typing import Sequence
from anndata import AnnData
import logging


def find_reference_candidates(
    adata: AnnData,
    reference_key: str = "cell_type",
    top_n: int = 5
) -> list[str]:
    """
    Identify candidate reference cell types based on lowest average variance.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    reference_key : str
        Column in adata.obs containing cell type labels.
    top_n : int
        Number of top reference candidates to return.

    Returns
    -------
    List of cell type names with lowest average expression variance.
    """
    categories = adata.obs[reference_key].unique()
    results = []
    for cat in categories:
        subset = adata[adata.obs[reference_key] == cat].X
        if scipy.sparse.issparse(subset):
            subset = subset.toarray()
        avg_var = np.var(subset, axis=0).mean()
        results.append((cat, avg_var))
    results.sort(key=lambda x: x[1])

    logging.info(f"Top {top_n} reference candidates based on variance:")
    for cat, var in results[:top_n]:
        logging.info(f"{cat}: avg variance = {var:.4f}")
    
    return [cat for cat, _ in results[:top_n]]


def _get_reference(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: None | str | Sequence[str],
    reference: np.ndarray | None,
) -> np.ndarray:
    """
    Extract reference gene expression profile.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    reference_key : str or None
        Column in adata.obs with reference labels.
    reference_cat : str, list of str, or None
        Category/categories in `reference_key` that indicate reference cells.
    reference : np.ndarray or None
        Precomputed reference matrix (overrides other inputs).

    Returns
    -------
    np.ndarray
        2D array of reference expression (n_refs x n_genes).
    """
    if reference is None:
        if reference_key is None or reference_cat is None:
            logging.warning(
                "Using mean of all cells as reference. For better results, "
                "provide either `reference`, or both `reference_key` and `reference_cat`."
            )
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
                    f"adata.obs[{reference_key!r}]: {reference_cat[~reference_cat_in_obs]}"
                )

            reference = np.vstack([
                np.mean(adata.X[obs_col.values == cat, :], axis=0)
                for cat in reference_cat
            ])

    if reference.ndim == 1:
        reference = reference[np.newaxis, :]

    if reference.shape[1] != adata.shape[1]:
        raise ValueError("Reference must match the number of genes in AnnData.")

    return reference

