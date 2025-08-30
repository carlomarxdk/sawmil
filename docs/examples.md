## Examples with the Dummy Data

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