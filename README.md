# **windowCNV**

**windowCNV** is a window-based tool for detecting copy number alterations (CNAs) from single-cell RNA-seq data. Inspired by [infercnvpy](https://github.com/icbi-lab/infercnvpy), it extends its functionality with flexible CNA simulation, inference, and evaluation. The package supports cell type–aware analysis and event-level performance metrics.

> **Note:** This package is experimental. CNA classification accuracy in real-world datasets may be limited. Feedback and contributions are welcome.

---

## Features

- Simulate CNAs globally or per cell type
- Infer CNAs from gene expression using customizable smoothing windows
- Assign CNA events to individual cells
- Evaluate **precision**, **recall**, and **F1 score** at the **event level**
- Visualize CNAs with heatmaps and summary tables

---

## Installation

We recommend using a dedicated conda environment:

```python
conda create -n windowcnv python=3.10
conda activate windowcnv
```

Then install windowCNV and its required dependencies:

```python
pip install infercnvpy scanpy matplotlib pandas
pip install git+https://github.com/Li-Jiabei/windowCNV.git
```

---

## Getting Started

Import the required packages:

```python
import numpy as np
import pandas as pd
import scanpy as sc
import infercnvpy as cnv
import matplotlib.pyplot as plt
import warnings
from collections.abc import Sequence

import windowCNV as wcnv
```
---

## Example Notebooks and Data

You can explore how to use **windowCNV** in the following notebooks:

### Simulated CNAs (Benchmarking & Validation)
* [**Original infercnvpy usage**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202A%20original%20infercnvpy.ipynb): Shows the baseline workflow using `infercnvpy`, enhanced with our new plotting and evaluation functions.
* [**windowCNV implementation**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202A%20WindowCNV.ipynb): Demonstrates the core `windowCNV` pipeline and comparison with `infercnvpy`.

These notebooks use the benchmarking dataset:
[PBMC\_simulated\_cnas\_041025.h5ad](https://jhu.instructure.com/files/13967706/download?download_frd=1)

---

* [**CNA simulation and windowCNV application**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202B.ipynb): Shows how to simulate CNAs and apply `windowCNV` inference.

This notebook uses the dataset:
[pbmc\_10k\_v3\_filtered\_feature\_bc\_matrix.h5](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5)

> Note: Many real-world datasets (including the one above) lack chromosome and genomic position annotations in `AnnData.var`.
> To address this, we provide a helper function for automatic annotation. The usage is shown in the notebook. However, you must supply a gene annotation file.

In our example, we use the following reference file:
[mart\_export\_GRCh38.p14.txt](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/data/mart_export_GRCh38.p14.txt)
This file contains:

* Gene stable ID
* Gene name
* Chromosome/scaffold name
* Gene start (bp)
* Gene end (bp)

The file was generated using [Ensembl BioMart](https://www.ensembl.org/biomart/martview/), which allows easy download of such annotations for genome build.

---

### PSC scRNA-seq data with previously reported PSC CNAs
* [**windowCNV on PBMC-4k with CAS CNA Labels**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%203%20PBMC_4k_10x.ipynb): Applies windowCNV to a 10x PBMC dataset with CAS-based high-confidence labels.

This notebook uses the dataset: [SCP2745_high_conf_CAS_cell_types.h5ad](https://singlecell.broadinstitute.org/single_cell/study/SCP2745/pbmc-4k-10x-h5ad-with-cas-results#study-download)

* [**windowCNV on TNBC iPSC-derived scRNA-seq data**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%203%20WindowCNV_TNBC.ipynb): Applies windowCNV to a triple-negative breast cancer (TNBC) iPSC dataset to identify large-scale CNA patterns across annotated cell types.

This notebook uses the dataset: [GSM4476486_combined_UMIcount_CellTypes_TNBC1.txt.gz](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4476486&format=file&file=GSM4476486%5Fcombined%5FUMIcount%5FCellTypes%5FTNBC1%2Etxt%2Egz)

* [**Krishna's**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%203%20WindowCNV_TCell.ipynb): description

This notebook uses the dataset: [GSM7744300_GUIDEvsNT_CHR14_RESULTS.txt](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM7744300&format=file&file=GSM7744300%5FGUIDEvsNT%5FCHR14%5FRESULTS%2Etxt%2Egz) and [gencode.v38.annotation.gtf](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz)
