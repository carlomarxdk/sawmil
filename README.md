[![PyPI version](https://img.shields.io/pypi/v/sawmil.svg)](https://pypi.org/project/sawmil/)
![Python versions](https://img.shields.io/pypi/pyversions/sawmil.svg)
![Wheel](https://img.shields.io/pypi/wheel/sawmil.svg)
![License](https://img.shields.io/pypi/l/sawmil.svg)
[![DOI](https://zenodo.org/badge/1046623935.svg)](https://doi.org/10.5281/zenodo.16990499)

# Sparse Multiple-Instance Learning in Python

MIL models based on the Support Vector Machines (NSK, sMIL, sAwMIL).
Inspired by the outdated [misvm](https://github.com/garydoranjr/misvm) package.

**Note**: This is an alpha version.

## Implemented Models

### Normalized Set Kernels (`NSK`)

> Gärtner, Thomas, Peter A. Flach, Adam Kowalczyk, and Alex J. Smola. [Multi-instance kernels](https://dl.acm.org/doi/10.5555/645531.656014). Proceedings of the 19th International Conference on Machine Learning (2002).

### Sparse MIL (`sMIL`)

> Bunescu, Razvan C., and Raymond J. Mooney. [Multiple instance learning for sparse positive bags](https://dl.acm.org/doi/10.1145/1273496.1273510). Proceedings of the 24th International Conference on Machine Learning (2007).

### Sparse Aware MIL (`sAwMIL`)

Classifier used in [trilemma-of-truth](https://github.com/carlomarxdk/trilemma-of-truth):
> Savcisens, Germans, and Tina Eliassi-Rad. [The Trilemma of Truth in Large Language Models](https://arxiv.org/abs/2506.23921). arXiv preprint arXiv:2506.23921 (2025).

## Installation

```bash
pip install sawmil
```

## Requirements

```bash
numpy>=1.22
scikit-learn>=1.7.0
gurobipy>=12.0.3
python>=11.0 # recommended: >=12.3
```

At this point, `sawmil` package works only with the [Gurobi](https://gurobi.com) optimizer. You need to obtain a academic/commercial license to use it. We plan to add implementations with other solvers.

## Quick start

### 1. Generate dummy data

``` python
from dataset import make_complex_bags
import numpy as np
rng = np.random.default_rng(0)

ds = make_complex_bags(
    n_pos=300, n_neg=100, inst_per_bag=(5, 15), d=2,
    pos_centers=((+2,+1), (+4,+3)),
    neg_centers=((-1.5,-1.0), (-3.0,+0.5)),
    pos_scales=((2.0, 0.6), (1.2, 0.8)),
    neg_scales=((1.5, 0.5), (2.5, 0.9)),
    pos_intra_rate=(0.25, 0.85),
    ensure_pos_in_every_pos_bag=True,
    neg_pos_noise_rate=(0.00, 0.05),
    pos_neg_noise_rate=(0.00, 0.20),
    outlier_rate=0.1,
    outlier_scale=8.0,
    random_state=42,
)
```

### 2. NSK with RBF Kernel

**Load a kernel:**

```python
from sawmil.kernels import get_kernel
from sawmil.bag_kernels import make_bag_kernel
k = get_kernel("rbf", gamma=0.5) # base (single-instance kernel)
bag_k  = make_bag_kernel(k, use_intra_labels=False) # convert single-instance kernel to bagged kernel
```

**Fit NSK Model:**

```python
from sawmil.nsk import NSK

clf = NSK(C=0.1, bag_kernel=bag_k, scale_C=True, tol=1e-8, verbose=False).fit(ds, None)
print("Train acc:", clf.score(ds, np.array([b.y for b in ds.bags])))
```

### 3. Fit sMIL Model with Linear Kernel

```python
from src.sawmil.smil import sMIL

k = get_kernel("linear", normalizer="none") # base (single-instance kernel)
bag_k  = make_bag_kernel(Linear(), normalizer="none", use_intra_labels=False)
clf = sMIL(C=0.1, bag_kernel=bag_k, scale_C=True, tol=1e-6, verbose=False).fit(ds, None)

print("Train acc:", clf.score(ds, np.array([1 if b.y > 0 else -1 for b in ds.bags])))
```

See more examples in the [`example.ipynb`](https://github.com/carlomarxdk/sawmil/blob/main/example.ipynb) notebook.

## Citation

If you use `sawmil` package in academic work, please cite:

Savcisens, G. & Eliassi-Rad, T. *sAwMIL: Python package for Sparse Multiple-Instance Learning* (2025).

```bibtex
@software{savcisens2025sawmil,
  author = {Savcisens, Germans and Eliassi-Rad, Tina},
  title = {sAwMIL: Python package for Sparse Multiple-Instance Learning},
  year = {2025},
  doi = {10.5281/zenodo.16990499},
  url = {https://github.com/carlomarxdk/sawmil}
}
```

If you want to reference a specific version of the package, find the [correct DOI here](https://doi.org/10.5281/zenodo.16990499).
