# **windowCNV**

**windowCNV** is a window-based tool for detecting copy number variations (CNVs) from single-cell RNA-seq data. Inspired by [infercnvpy](https://github.com/icbi-lab/infercnvpy), it extends its functionality with flexible CNV simulation, inference, and evaluation. The package supports cell type–aware analysis and event-level performance metrics.

> **Note:** This package is experimental. CNV classification accuracy in real-world datasets may be limited. Feedback and contributions are welcome.

---

## Features

- Simulate CNVs globally or per cell type
- Infer CNVs from gene expression using customizable smoothing windows
- Assign CNV events to individual cells
- Evaluate **precision**, **recall**, and **F1 score** at the **event level**
- Visualize CNVs with heatmaps and summary tables

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

* [**Original `infercnvpy` usage**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202A%20original%20infercnvpy.ipynb): Shows the baseline workflow using `infercnvpy`, enhanced with our new plotting and evaluation functions.
* [**windowCNV implementation**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202A%20WindowCNV.ipynb): Demonstrates the core `windowCNV` pipeline and comparison with `infercnvpy`.

These notebooks use the benchmarking dataset:
**[PBMC\_simulated\_cnas\_041025.h5ad](https://jhu.instructure.com/files/13967706/download?download_frd=1)**

---

* [**CNV simulation and windowCNV application**](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202B%20CNV%20simulation%20and%20WindowCNV.ipynb) (‼️needs link): Shows how to simulate CNVs and apply `windowCNV` inference.

This notebook uses the dataset:
**[pbmc\_10k\_v3\_filtered\_feature\_bc\_matrix.h5](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5)**

> Note: Many real-world datasets (including the one above) lack chromosome and genomic position annotations in `AnnData.var`.
> To address this, we provide a helper function for automatic annotation. The usage is shown in the notebook. However, you must supply a gene annotation file.

In our example, we use the following reference file:
**[mart\_export\_GRCh38.p14.txt](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/data/mart_export_GRCh38.p14.txt)**
This file contains:

* Gene stable ID
* Gene name
* Chromosome/scaffold name
* Gene start (bp)
* Gene end (bp)

The file was generated using [Ensembl BioMart](https://www.ensembl.org/biomart/martview/), which allows easy download of such annotations for genome build.
