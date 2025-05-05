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

You can explore the usage in these example notebooks:

* [Original infercnvpy usage with new plotting functions and performance metrics](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202A%20original%20infercnvpy.ipynb)
* [Fundemental windowCNV implementation](https://github.com/Li-Jiabei/windowCNV/blob/main/windowCNV/tests/Task%202A%20WindowCNV.ipynb)
* [CNV simulation and windowCNV implementation]
