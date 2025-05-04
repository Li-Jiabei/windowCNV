"""
simulate.py

Simulate copy number alterations (CNAs) in single-cell RNA-seq data.
Includes global and cell-type-specific simulation, overlap checking,
and summarization utilities.
"""

import numpy as np
import pandas as pd
import random
import scipy.sparse
from collections import defaultdict
from scipy.sparse import issparse

# --- CNA generation utilities ---

def random_split_cnas(n_total_cnas, random_seed=555):
    """
    Randomly split a total number of CNAs into small categories: gain, heterozygous deletion, homozygous deletion.

    Parameters:
    -----------
    n_total_cnas : int
        Total number of CNAs to split.

    random_seed : int, default=555
        Random seed for reproducibility.

    Returns:
    --------
    n_gain : int
        Number of gain CNAs (copy number = 4).

    n_hetero_del : int
        Number of heterozygous deletion CNAs (copy number = 1).

    n_homo_del : int
        Number of homozygous deletion CNAs (copy number = 0).
    """
    np.random.seed(random_seed)
    splits = np.random.multinomial(n_total_cnas, [1/3, 1/3, 1/3])
    return splits[0], splits[1], splits[2]

def check_overlap(existing_cnas, chrom, start, end):
    """
    Check whether a new candidate CNA region overlaps with any existing CNA region.

    Parameters:
    -----------
    existing_cnas : list of tuples
        Existing CNA regions, each as (chromosome, start, end).

    chrom : str
        Chromosome of the candidate CNA.

    start : int
        Start position of the candidate CNA.

    end : int
        End position of the candidate CNA.

    Returns:
    --------
    bool
        True if overlap exists, False otherwise.
    """
    for ex_chrom, ex_start, ex_end in existing_cnas:
        if chrom == ex_chrom and not (end < ex_start or start > ex_end):
            return True
    return False

# --- Core simulation functions ---

def simulate_cnas_basic(adata, n_gain, n_hetero_del, n_homo_del, size_ranges=None, random_seed=555, cna_effects=None):
    """
    Simulate CNAs (copy number alterations) globally without considering cell types.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing gene expression data.

    n_gain : int
        Number of gain CNAs to simulate (copy number = 4).

    n_hetero_del : int
        Number of heterozygous deletion CNAs to simulate (copy number = 1).

    n_homo_del : int
        Number of homozygous deletion CNAs to simulate (copy number = 0).

    size_ranges : dict or None, default=None
        Size ranges (in base pairs) for small, medium, large CNAs. If None, uses default ranges.

    random_seed : int, default=555
        Random seed for reproducibility.

    cna_effects : dict or None, default=None
        Multiplication factors for gain, heterozygous deletion, and homozygous deletion.
        Example: {'gain': 2, 'hetero_del': 0.5, 'homo_del': 0}.

    Returns:
    --------
    adata : AnnData
        Updated AnnData with simulated CNAs stored in `adata.obs['simulated_cnvs']`.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    if size_ranges is None:
        size_ranges = {'small': (5e5, 1e6), 'medium': (3e6, 1e7), 'large': (2e7, 5e7)}

    if cna_effects is None:
        cna_effects = {'gain': 2, 'hetero_del': 0.5, 'homo_del': 0}

    chromosomes_allowed = [str(i) for i in range(1, 23)] + ['X', 'Y']

    valid_genes = adata.var.dropna(subset=['chromosome', 'start', 'end'])
    valid_genes = valid_genes[valid_genes['chromosome'].isin(chromosomes_allowed)]

    simulated_cnas = []
    existing_cnas = []
    attempts = 0
    cna_types_list = ['gain'] * n_gain + ['hetero_del'] * n_hetero_del + ['homo_del'] * n_homo_del
    random.shuffle(cna_types_list)
    n_total_cnas = n_gain + n_hetero_del + n_homo_del

    while len(simulated_cnas) < n_total_cnas and attempts < 1000:
        size_label = 'small' if len(simulated_cnas) < n_total_cnas/3 else ('medium' if len(simulated_cnas) < 2*n_total_cnas/3 else 'large')

        chr_selected = np.random.choice(chromosomes_allowed)
        chr_genes = valid_genes[valid_genes['chromosome'] == chr_selected]
        if chr_genes.empty:
            attempts += 1
            continue

        start_min, start_max = chr_genes['start'].min(), chr_genes['end'].max()
        size_min, size_max = size_ranges[size_label]
        if start_max - start_min < size_min:
            attempts += 1
            continue

        start_pos = np.random.randint(start_min, start_max - size_min)
        end_pos = start_pos + np.random.randint(size_min, size_max)

        if check_overlap(existing_cnas, chr_selected, start_pos, end_pos):
            attempts += 1
            continue

        cna_type = cna_types_list.pop()
        label = f"{chr_selected}:{start_pos}-{end_pos} ({'CN 4' if cna_type == 'gain' else ('CN 1' if cna_type == 'hetero_del' else 'CN 0')})"
        simulated_cnas.append({'chr': chr_selected, 'start': start_pos, 'end': end_pos, 'genes': chr_genes.index.tolist(), 'type': cna_type, 'label': label})
        existing_cnas.append((chr_selected, start_pos, end_pos))
        attempts = 0

    if len(simulated_cnas) < n_total_cnas:
        print(f"Warning: Only generated {len(simulated_cnas)} CNAs instead of {n_total_cnas}")

    simulated_labels = pd.Series('', index=adata.obs_names)
    cell_to_idx = {cell: i for i, cell in enumerate(adata.obs_names)}
    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}

    for cna in simulated_cnas:
        affected_cells = adata.obs_names[np.random.rand(adata.n_obs) < 0.3]
        affected_cells_idx = [cell_to_idx[cell] for cell in affected_cells if cell in cell_to_idx]
        affected_genes_idx = [gene_to_idx[gene] for gene in cna['genes'] if gene in gene_to_idx]
        if not affected_cells_idx or not affected_genes_idx:
            continue
        sub_X = adata.X[affected_cells_idx, :][:, affected_genes_idx]
        factor = cna_effects[cna['type']]
        
        if isinstance(adata.X, scipy.sparse.spmatrix):
            # Convert to LIL format for efficient row assignment
            adata.X = adata.X.tolil()
            for i, cell_idx in enumerate(affected_cells_idx):
                adata.X[cell_idx, affected_genes_idx] = adata.X[cell_idx, affected_genes_idx] * factor
            adata.X = adata.X.tocsr()  # Convert back to CSR after modification

        else:
            adata.X[np.ix_(affected_cells_idx, affected_genes_idx)] = \
                (sub_X if isinstance(sub_X, np.ndarray) else sub_X.toarray()) * factor

        for cell in affected_cells:
            simulated_labels[cell] += ', ' + cna['label'] if simulated_labels[cell] else cna['label']

    adata.obs['simulated_cnvs'] = simulated_labels.astype('category')
    return adata

# --- Cell-type-specific simulation ---

def simulate_cnas_by_celltype(adata, celltype_col=None, celltype_cna_counts=None, size_ranges=None, random_seed=555, cna_effects=None):
    """
    Simulate CNAs separately for each cell type.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing gene expression data.

    celltype_col : str or None
        Column name in adata.obs that specifies cell type.

    celltype_cna_counts : dict or None
        Dictionary mapping cell type names to the number of CNAs to simulate for that type.

    size_ranges : dict or None, default=None
        Size ranges (in base pairs) for small, medium, large CNAs. If None, uses default.

    random_seed : int, default=555
        Random seed for reproducibility.

    cna_effects : dict or None, default=None
        Multiplication factors for different CNA types.

    Returns:
    --------
    adata : AnnData
        Updated AnnData with simulated CNAs stored in `adata.obs['simulated_cnvs']`.
    """
    if celltype_col is None or celltype_cna_counts is None:
        print("No cell type input, simulating globally...")
        return simulate_cnas_basic(adata, *random_split_cnas(30, random_seed), size_ranges=size_ranges, random_seed=random_seed, cna_effects=cna_effects)

    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{celltype_col}' not found in adata.obs.")

    np.random.seed(random_seed)
    random.seed(random_seed)

    simulated_labels = pd.Series('', index=adata.obs_names)

    for celltype, n_total_cnas in celltype_cna_counts.items():
        cells_in_type = adata.obs_names[adata.obs[celltype_col] == celltype]
        if len(cells_in_type) == 0 or n_total_cnas == 0:
            continue
        n_gain, n_hetero_del, n_homo_del = random_split_cnas(n_total_cnas, random_seed=random.randint(0, 99999))
        
        sub_adata = adata[cells_in_type].copy()
        sub_adata = simulate_cnas_basic(
            sub_adata,
            n_gain,
            n_hetero_del,
            n_homo_del,
            size_ranges=size_ranges,
            random_seed=random.randint(0, 99999),
            cna_effects=cna_effects
        )
        
        # Safely copy simulated expression back into original AnnData
        adata_subset = adata[cells_in_type, :].copy()
        adata_subset.X = sub_adata.X.copy() if not issparse(sub_adata.X) else sub_adata.X
        adata[cells_in_type, :] = adata_subset
        
        # Copy simulated CNV labels
        simulated_labels.loc[cells_in_type] = sub_adata.obs['simulated_cnvs']

    adata.obs['simulated_cnvs'] = simulated_labels.astype('category')
    return adata

# --- CNA summary utilities ---

def summarize_cna_regions(adata):
    """
    Summarize all unique simulated CNA regions from AnnData object.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing simulated CNAs.

    Returns:
    --------
    result_df : pandas.DataFrame
        DataFrame listing chromosome, start, end, and CNA label for each simulated region.
    """
    if 'simulated_cnvs' not in adata.obs.columns:
        raise ValueError("No 'simulated_cnvs' found in adata.obs.")

    cna_labels = set()
    for labels in adata.obs['simulated_cnvs']:
        if not labels:
            continue
        for label in labels.split(', '):
            cna_labels.add(label)

    records = []
    for label in cna_labels:
        chrom_pos, cna_info = label.split(' (')
        chrom, coords = chrom_pos.split(':')
        start, end = coords.split('-')
        start, end = int(start), int(end)
        cna_info = cna_info.replace(')', '')
        records.append({'chromosome': chrom, 'start': start, 'end': end, 'cna_label': cna_info})

    if not records:
        print("No CNA regions found.")
        return pd.DataFrame()

    result_df = pd.DataFrame(records)
    return result_df.sort_values(['chromosome', 'start']).reset_index(drop=True)

def print_celltype_to_cnv_chromosomes(adata, celltype_col='cell_type', cnv_col='simulated_cnvs'):
    """
    Print each cell type and the list of chromosomes where it has simulated CNAs, in a compact format.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing simulated CNA information.

    celltype_col : str, default='cell_type'
        Column name in adata.obs that stores the cell type annotation.

    cnv_col : str, default='simulated_cnvs'
        Column name in adata.obs that stores the simulated CNV labels.

    Returns:
    --------
    None
        Prints the mapping of cell types to the list of chromosomes with CNAs.
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{celltype_col}' not found in adata.obs.")

    if cnv_col not in adata.obs.columns:
        raise ValueError(f"CNV column '{cnv_col}' not found in adata.obs.")

    celltype_chr_cnv = defaultdict(set)

    for idx, row in adata.obs.iterrows():
        cell_type = row[celltype_col]
        cnv_labels = row[cnv_col]
        if not cnv_labels:
            continue
        for label in cnv_labels.split(', '):
            chrom = label.split(':')[0]
            celltype_chr_cnv[cell_type].add(chrom)

    for cell_type, chromosomes in sorted(celltype_chr_cnv.items()):
        chrom_list = sorted(chromosomes, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
        print(f"{cell_type}: {chrom_list}")

def map_cnv_status_by_celltype(
    adata,
    celltype_key='cell_type',
    cnv_truth_key='simulated_cnvs'
):
    """
    Map each cell type to the set of chromosomes with CNVs, including gain or loss direction.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    celltype_key : str
        Key in adata.obs containing cell type labels.
    cnv_truth_key : str
        Key in adata.obs containing ground truth CNV annotations.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: ['Cell Type', 'Chromosome', 'Gain', 'Loss']
    """
    import pandas as pd  # Add at the top of the file if not already there

    records = []
    for cell_type in adata.obs[celltype_key].unique():
        subset = adata.obs[adata.obs[celltype_key] == cell_type]
        chrom_cnv = {}

        for annotation in subset[cnv_truth_key].dropna():
            events = str(annotation).split(',')
            for event in events:
                event = event.strip()
                if event:
                    try:
                        chrom_part, cn_part = event.split('(')
                        chrom = chrom_part.split(':')[0].strip()
                        cn = int(cn_part.strip(' CN)').strip())
                        if not chrom.startswith('chr'):
                            chrom = f"chr{chrom}"
                        if chrom not in chrom_cnv:
                            chrom_cnv[chrom] = {'gain': False, 'loss': False}
                        if cn > 2:
                            chrom_cnv[chrom]['gain'] = True
                        elif cn < 2:
                            chrom_cnv[chrom]['loss'] = True
                    except Exception as e:
                        print(f"Warning: failed to parse event '{event}': {e}")
                        continue

        for chrom, status in chrom_cnv.items():
            records.append({
                'Cell Type': cell_type,
                'Chromosome': chrom,
                'Gain': status['gain'],
                'Loss': status['loss']
            })

    return pd.DataFrame(records)

