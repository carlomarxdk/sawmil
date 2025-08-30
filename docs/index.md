# Welcome! 

**This package is in the alpha stage of testings.**

`sAwMIL` provides  Multiple-Instance Learning (MIL) models built on support vector machines. It is inspired by the outdated [misvm](https://github.com/garydoranjr/misvm) package.

1. **SVM** (single-instance baseline)
2. **NSK** (Normalized Set Kernel)
   > Gärtner, Thomas, Peter A. Flach, Adam Kowalczyk, and Alex J. Smola. [Multi-instance kernels](https://dl.acm.org/doi/10.5555/645531.656014). Proceedings of the 19th International Conference on Machine Learning (2002).
3. **sMIL** (Sparse MIL)
   > Bunescu, Razvan C., and Raymond J. Mooney. [Multiple instance learning for sparse positive bags](https://dl.acm.org/doi/10.1145/1273496.1273510). Proceedings of the 24th International Conference on Machine Learning (2007).
4. **sAwMIL** (Sparse-Aware MIL; two-stage: sMIL → instance SVM)

    Classifier used in [trilemma-of-truth](https://github.com/carlomarxdk/trilemma-of-truth):
    > Savcisens, Germans, and Tina Eliassi-Rad. [The Trilemma of Truth in Large Language Models](https://arxiv.org/abs/2506.23921). arXiv preprint arXiv:2506.23921 (2025).

Our package supports two **QP backends**: [Gurobi](https://gurobi.com) and [OSQP](https://osqp.org/).