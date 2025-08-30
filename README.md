# Sparse Multiple-Instance Learning in Python

[![PyPI version](https://img.shields.io/pypi/v/sawmil.svg)](https://pypi.org/project/sawmil/)
![Python versions](https://img.shields.io/pypi/pyversions/sawmil.svg)
![Wheel](https://img.shields.io/pypi/wheel/sawmil.svg)
![License](https://img.shields.io/pypi/l/sawmil.svg)
[![DOI](https://zenodo.org/badge/1046623935.svg)](https://doi.org/10.5281/zenodo.16990499)
[![Docs](https://img.shields.io/badge/docs-latest-purple?logo=mkdocs)](https://carlomarxdk.github.io/sawmil/)


> [!Warning]
>
> Note: It is an **alpha** version of the package.  This is an alpha version (see [Documentation](https://carlomarxdk.github.io/sawmil/)).

MIL models based on the Support Vector Machines (NSK, sMIL, sAwMIL).
Inspired by the outdated [misvm](https://github.com/garydoranjr/misvm) package.

## Documentation

Refer to the [Initial Documentation](https://carlomarxdk.github.io/sawmil/).

## Implemented Models

### Normalized Set Kernels (`NSK`)

> Gärtner, Thomas, Peter A. Flach, Adam Kowalczyk, and Alex J. Smola. [Multi-instance kernels](https://dl.acm.org/doi/10.5555/645531.656014). Proceedings of the 19th International Conference on Machine Learning (2002).

### Sparse MIL (`sMIL`)

> Bunescu, Razvan C., and Raymond J. Mooney. [Multiple instance learning for sparse positive bags](https://dl.acm.org/doi/10.1145/1273496.1273510). Proceedings of the 24th International Conference on Machine Learning (2007).

### Sparse Aware MIL (`sAwMIL`)

Classifier used in [trilemma-of-truth](https://github.com/carlomarxdk/trilemma-of-truth):
> Savcisens, Germans, and Tina Eliassi-Rad. [The Trilemma of Truth in Large Language Models](https://arxiv.org/abs/2506.23921). arXiv preprint arXiv:2506.23921 (2025).

---

## Installation

`sawmil` supports two QP backends: [Gurobi](https://gurobi.com) and [OSQP](https://osqp.org/).
By default, the base package installs **without** any solver; pick one (or both) via extras.

### Base package (no solver)

```bash
pip install sawmil
# it installs numpy>=1.22 and scikit-learn>=1.7.0
```

### Option 1 — Gurobi backend

> Gurobi is commercial software. You’ll need a valid license (academic or commercial), refer to the [official website](https://gurobi.com).

```bash
pip install "sawmil[gurobi]"
# in additionl to the base packages, it install gurobi>12.0.3
```

### Option 2 — OSQP backend

```bash
pip install "sawmil[osqp]"
# in additionl to the base packages, it installs osqp>=1.0.4 and scipy
```

### Option 3 — All supported solvers

```bash
pip install "sawmil[full]"
```

### Picking the solver in code

```python
from sawmil import SVM, RBF

k = RBF(gamma = 0.1)
# solver= "osqp" (default is "gurobi")
# SVM is for single-instances 
clf = SVM(C=1.0, 
          kernel=k, 
          solver="osqp").fit(X, y)
```

## Quick start

### 1. Generate Dummy Data

``` python
from sawmil.data import generate_dummy_bags
import numpy as np
rng = np.random.default_rng(0)

ds = generate_dummy_bags(
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

### 2. Fit `NSK` with RBF Kernel

**Load a kernel:**

```python
from sawmil.kernels import get_kernel, RBF
k1 = get_kernel("rbf", gamma=0.1)
k2 = RBF(gamma=0.1)
# k1 == k2

```

**Fit NSK Model:**

```python
from sawmil.nsk import NSK

clf = NSK(C=1, kernel=k, 
          # bag kernel settings
          normalizer='average',
          # solver params
          scale_C=True, 
          tol=1e-8, 
          verbose=False).fit(ds, None)
y = ds.y
print("Train acc:", clf.score(ds, y))
```

### 3. Fit `sMIL` Model with Linear Kernel

```python
from sawmil.smil import sMIL

k = get_kernel("linear") # base (single-instance kernel)
clf = sMIL(C=0.1, 
           kernel=k, 
           scale_C=True, 
           tol=1e-8, 
           verbose=False).fit(ds, None)
```

See more examples in the [`example.ipynb`](https://github.com/carlomarxdk/sawmil/blob/main/example.ipynb) notebook.

### 4. Fit `sAwMIL` with Combined Kernels

```python
from sawmil.kernels import Product, Polynomial, Linear, RBF, Sum, Scale
from sawmil.sawmil import sAwMIL

k = Sum(Linear(), 
        Scale(0.5, 
              Product(Polynomial(degree=2), RBF(gamma=1.0))))

clf = sAwMIL(C=0.1, 
             kernel=k,
             solver="gurobi", 
             eta=0.95) # here eta is high, since all items in the bag are relevant
clf.fit(ds)
print("Train acc:", clf.score(ds, ds.y))
```

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
