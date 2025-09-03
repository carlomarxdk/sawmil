# Welcome! 

**This package is in the alpha stage of testings.**

`sAwMIL` provides  Multiple-Instance Learning (MIL) models built on support vector machines. It is inspired by the outdated [misvm](https://github.com/garydoranjr/misvm) package. Our package supports two **QP backends**: [Gurobi](https://gurobi.com) and [OSQP](https://osqp.org/).

## Single-Instance SVMs

1. **SVM** (Single-Instance SVM)

## Multiple-Instance SVMs

1. **NSK** (Normalized Set Kernel) [@gartner2002multi]
   See the exapliner on [NSK][sawmil.nsk.NSK] class for details.
2. **sMIL** (Sparse MIL)
   > Bunescu, Razvan C., and Raymond J. Mooney. [Multiple instance learning for sparse positive bags](https://dl.acm.org/doi/10.1145/1273496.1273510). Proceedings of the 24th International Conference on Machine Learning (2007).
3. **sAwMIL** (Sparse-Aware MIL; two-stage: sMIL → instance SVM)
    Classifier used in [trilemma-of-truth](https://github.com/carlomarxdk/trilemma-of-truth):
    > Savcisens, Germans, and Tina Eliassi-Rad. [The Trilemma of Truth in Large Language Models](https://arxiv.org/abs/2506.23921). arXiv preprint arXiv:2506.23921 (2025).

