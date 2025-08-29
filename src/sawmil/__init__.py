__all__ = [
    "SVM", "NSK", "sMIL", "sAwMIL",
    "BaseKernel", "Linear", "RBF", "Polynomial", "Sigmoid",
    "Normalize", "Scale", "Sum", "Product",
    "WeightedMeanBagKernel", "make_bag_kernel",
    "Bag", "BagDataset",
]

__version__ = "0.1.1"

from .svm import SVM
from .nsk import NSK
from .smil import sMIL
from .sawmil import sAwMIL

from .kernels import (
    BaseKernel, Linear, RBF, Polynomial, Sigmoid,
    Normalize, Scale, Sum, Product,
)

from .bag_kernels import WeightedMeanBagKernel, make_bag_kernel
from .bag import Bag, BagDataset