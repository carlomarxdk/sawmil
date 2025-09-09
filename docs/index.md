# Welcome to sawmil's documentation!

**This package is in the alpha stage of testing.**

`sAwMIL` (**S**parse **Aw**are **M**ultiple-**I**nstance **L**earning) is an open-source Python library providing a collection of Support Vector Machine (SVM) classifiers for multiple-instance learning (MIL). It builds upon ideas from the earlier [misvm](https://github.com/garydoranjr/misvm) package, adapting it for the latest Python version, as well as introducing new models.

In **Single-Instance Learning** (SIL), the dataset consists of pairs of an instance and a label:

\[
\langle \boldsymbol{x}_i, y_i \rangle \text{ , where } \boldsymbol{x}_i \in \mathbb{R}^{d} \text{ and } y_i \in \mathcal{Y}.
\]

In binary settings, the label is \( y \in \{0,1\} \).
To solve this problem, we can use a standard [SVM][sawmil.svm.SVM] model.

In **Multiple-Instance Learning** (MIL), the dataset consists of *bags* of instances paired with a single bag-level label:

\[
\langle \boldsymbol{X}_i, y_i \rangle \text{ , where } \boldsymbol{X}_i = \{ \boldsymbol{x}_{1}, \boldsymbol{x}_{2}, ..., \boldsymbol{x}_{n_i} \}, \boldsymbol{x}_j \in \mathbb{R}^{d} \text{ and } y_i \in \mathcal{Y}.
\]

To solve this problem, we can use [NSK][sawmil.nsk.NSK] or [sMIL][sawmil.smil.sMIL] models.

In some cases, each bag, along with the instances and a label, could contain a **intra-bag mask** that specifies which items are likely to contain the signal related to \(y\). In that case, we have a triplet of \( \langle \boldsymbol{X}_i, \boldsymbol{M}_i, y_i \rangle \), where

\[
 \boldsymbol{M}_i = \{m_1, m_1,... m_{n_i}\}, \text{ where } m_j \in \{0,1\}.
\]

To solve this problem, one can use the [sAwMIL][sawmil.sawmil.sAwMIL] model.

## Solvers

Our package supports three  **QP backends**: [Gurobi](https://gurobi.com),  [OSQP](https://osqp.org/) and [DAQP](https://darnstrom.github.io/daqp/).

## Models

### Single-Instance SVMs

Our package implements a simple [SVM][sawmil.svm.SVM] for the SIL setting.

### Multiple-Instance SVMs

The multiple-instance SVMs are adapted to work with the bags

1. **NSK** (Normalized Set Kernel) [@gartner2002multi]
2. **sMIL** (Sparse Multiple-Instance Learnig) [@bunescu2007multiple]
3. **sAwMIL** (Sparse Aware MIL) [@savcisens2025trilemma] is used in [trilemma-of-truth](https://github.com/carlomarxdk/trilemma-of-truth).
