__version__ = "0.1.12"

from .svm import SVM as SVM
from .nsk import NSK as NSK
from .smil import sMIL as sMIL
from .sawmil import sAwMIL as sAwMIL

from .kernels import (
    BaseKernel as BaseKernel, Linear as Linear, RBF as RBF,
    Polynomial as Polynomial, Sigmoid as Sigmoid,
    Normalize as Normalize, Scale as Scale, Sum as Sum, Product as Product,
)

from .bag_kernels import WeightedMeanBagKernel as WeightedMeanBagKernel, make_bag_kernel as make_bag_kernel
from .bag import Bag as Bag, BagDataset as BagDataset

__all__ = [
    "SVM", "NSK", "sMIL", "sAwMIL",
    "BaseKernel", "Linear", "RBF", "Polynomial", "Sigmoid",
    "Normalize", "Scale", "Sum", "Product",
    "WeightedMeanBagKernel", "make_bag_kernel",
    "Bag", "BagDataset",
]