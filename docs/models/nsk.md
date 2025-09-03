# Normalized Set Kernel

This approach is based on the method descibed in
> @gartner2002multi

## Simple Use

```python
from sawmil import NSK # from sawmil.nsk import NSK (equivalent)
from sawmil import Linear, RBF # from sawmil.kernels import Linear, RBF (equivalent)

# 1. Define a kernel 
kernel = Linear()

# 2. Specify the model
# NSK inherits the structure of the sklearn models
# even though you supply a single-instance kernel, the NSK object converts it to the bagged kernel
model = NSK(C = 0.1, kernel = kernel)

# 3. Train
model.train(bag_dataset, None)
```

::: sawmil.nsk.NSK

