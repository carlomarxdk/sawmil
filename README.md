# Multiple-Instance Support Vector Machines in Python

[![PyPI - Version](https://img.shields.io/pypi/v/sawmil)](https://pypi.org/project/sawmil/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sawmil)](https://pypi.org/project/sawmil/)
[![PyPI - Status](https://img.shields.io/pypi/status/sawmil)](https://pypi.org/project/sawmil/)
[![GitHub License](https://img.shields.io/github/license/carlomarxdk/sawmil)](https://github.com/carlomarxdk/sawmil/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/sawmil/badge/?version=latest)](https://sawmil.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/carlomarxdk/sawmil/graph/badge.svg?token=0UY8N6FH4Y)](https://codecov.io/github/carlomarxdk/sawmil)
[![CodeFactor](https://www.codefactor.io/repository/github/carlomarxdk/sawmil/badge/main)](https://www.codefactor.io/repository/github/carlomarxdk/sawmil/overview/main)
[![DOI](https://zenodo.org/badge/1046623935.svg)](https://doi.org/10.5281/zenodo.16990499)

`sAwMIL` (**S**parse **Aw**are **M**ultiple-**I**nstance **L**earning) is an open-source Python library providing a collection of Support Vector Machine (SVM) classifiers for multiple-instance learning (MIL). It builds upon ideas from the earlier [misvm](https://github.com/garydoranjr/misvm) package, adapting it for the latest Python version, as well as introducing new models.

In **Single-Instance Learning** (SIL), the dataset consists of pairs of an instance and a label:

$$
\langle \mathbf{x}_i, y_i \rangle \text{ , where } \mathbf{x}_i \in \mathbb{R}^{d} \text{ and } y_i \in \mathcal{Y}.
$$

In binary settings, the label is $y \in \{0,1\}$.
To solve this problem, we can use a standard [SVM](https://sawmil.readthedocs.io/en/latest/models/svm/) model.

In **Multiple-Instance Learning** (MIL), the dataset consists of *bags* of instances paired with a single bag-level label:

$$
\langle \mathbf{X}_i, y_i \rangle \text{ , where } \mathbf{X}_i = \{ \mathbf{x}_{1}, \mathbf{x}_{2}, ..., \mathbf{x}_{n_i} \}, \mathbf{x}_j \in \mathbb{R}^{d} \text{ and } y_i \in \mathcal{Y}.
$$

To solve this problem, we can use [NSK](https://sawmil.readthedocs.io/en/latest/models/nsk/) or [sMIL](https://sawmil.readthedocs.io/en/latest/models/sMIL/) models.

In some cases, each bag, along with the instances and a label, could contain a **intra-bag mask** that specifies which items are likely to contain the signal related to $y$. In that case, we have a triplet of $\langle \mathbf{X}_i, \mathbf{M}_i, y_i \rangle$, where

$$
 \mathbf{M}_i = \{m_1, m_1,... m_{n_i}\}, \text{ where } m_j \in \{0,1\}.
$$

To solve this problem, one can use the [sAwMIL](https://sawmil.readthedocs.io/en/latest/models/sAwMIL/) model.

## Installation

`sawmil` supports three QP backends:

* [Gurobi](https://gurobi.com)
* [OSQP](https://osqp.org/)
* [DAQP](https://darnstrom.github.io/daqp/)
  
By default, the base package installs **without** any solver; pick one (or both) via extras.

### Base package (no solver)

```bash
pip install sawmil
# it installs numpy>=1.22 and scikit-learn>=1.7.0
```

### Option 1: `Gurobi` backend

> Gurobi is commercial software. You’ll need a valid license (academic or commercial), refer to the [official website](https://gurobi.com).

```bash
pip install "sawmil[gurobi]"
# in additionl to the base packages, it install gurobi>12.0.3
```

### Option 2: `OSQP` backend

```bash
pip install "sawmil[osqp]"
# in additionl to the base packages, it installs osqp>=1.0.4 and scipy>=1.16.1
```

### Option 3: `DAQP` backend

```bash
pip install "sawmil[daqp]"
# in additionl to the base packages, it installs daqp>=0.5 and scipy>=1.16.1
```

### Option 4 — All supported solvers

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
