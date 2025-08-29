# Sparse Multiple-Instance Learning in Python

MIL models based on the Support Vector Machines (NSK, sMIL, sAwMIL).
Inspired by the outdated [misvm](https://github.com/garydoranjr/misvm) package.

**Note**: This is an alpha version.

## Installation

```bash
pip install sawmil
```

## Quick start

```python
from sawmil.svm import SVM

clf = SVM(kernel="linear")
clf.fit(X, y)
```

See `example.ipynb`.