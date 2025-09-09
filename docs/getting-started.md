# Installation

`sawmil` supports three QP backends:

* [Gurobi](https://gurobi.com)
* [OSQP](https://osqp.org/)
* [DAQP](https://darnstrom.github.io/daqp/)
  
By default, the base package installs **without** any solver; pick one (or both) via extras.

## Base package (no solver)

```bash
pip install sawmil
# it installs numpy>=1.22 and scikit-learn>=1.7.0
```

## Option 1: `Gurobi` backend

> Gurobi is commercial software. You’ll need a valid license (academic or commercial), refer to the [official website](https://gurobi.com).

```bash
pip install "sawmil[gurobi]"
# in additionl to the base packages, it install gurobi>12.0.3
```

## Option 2: `OSQP` backend

```bash
pip install "sawmil[osqp]"
# in additionl to the base packages, it installs osqp>=1.0.4 and scipy>=1.16.1
```

## Option 3: `DAQP` backend

```bash
pip install "sawmil[daqp]"
# in additionl to the base packages, it installs daqp>=0.5 and scipy>=1.16.1
```

## Option 4 — All supported solvers

```bash
pip install "sawmil[full]"
```

## Picking the solver in code

```python
from sawmil import SVM, RBF

k = RBF(gamma = 0.1)
# solver= "osqp" (default is "gurobi")
# SVM is for single-instances 
clf = SVM(C=1.0, 
          kernel=k, 
          solver="osqp").fit(X, y)
```