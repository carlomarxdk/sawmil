# tests/test_svm.py
import numpy as np
import pytest
from typing import Any
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef as mcc

# Prefer installed package; fall back to local src/ when running from repo
try:
    from sawmil.svm import SVM
    from sawmil.kernels import Linear, RBF, Polynomial, Sigmoid
except Exception:  # pragma: no cover
    from ..src.sawmil.svm import SVM  # type: ignore
    from ..src.sawmil.kernels import Linear, RBF, Polynomial, Sigmoid  # type: ignore


# ----------------- helpers -----------------

def _standardize(X):
    return StandardScaler().fit_transform(X)

def _linear_toy(n_per=25, sep=2.0, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(loc=[+sep, +sep], scale=noise, size=(n_per, 2))
    X_neg = rng.normal(loc=[-sep, -sep], scale=noise, size=(n_per, 2))
    X = np.vstack([X_pos, X_neg]).astype(float)
    # Intentionally 0/1 labels (not {-1, +1})
    y = np.hstack([np.ones(n_per), np.zeros(n_per)])
    return _standardize(X), y

def _moons(n=120, noise=0.2, seed=2):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return _standardize(X), y

def _make_kernel(name: str, params: dict):
    name = name.lower()
    if name == "linear":
        return Linear()
    if name == "rbf":
        return RBF(gamma=params.get("gamma"))
    if name == "poly" or name == "polynomial":
        return Polynomial(
            degree=params.get("degree", 3),
            gamma=params.get("gamma"),
            coef0=params.get("coef0", 0.0),
        )
    if name == "sigmoid":
        return Sigmoid(
            gamma=params.get("gamma"),
            coef0=params.get("coef0", 0.0),
        )
    raise ValueError(f"Unknown kernel: {name}")

def _mirror_sk_params(kernel_name, params):
    """Build sklearn SVC params parallel to ours."""
    out = dict(C=params.get("C", 1.0), kernel=kernel_name)
    for k in ("gamma", "degree", "coef0"):
        if k in params:
            out[k] = params[k]
    return out


# ----------------- solver fixture -----------------

@pytest.fixture(scope="function", params=["gurobi", "osqp"])
def solver_name(request):
    """Parametrize over solvers; skip if not installed."""
    name = request.param
    if name == "gurobi":
        pytest.importorskip("gurobipy", reason="Gurobi not installed")
    elif name == "osqp":
        pytest.importorskip("osqp", reason="OSQP not installed")
        pytest.importorskip("scipy", reason="OSQP requires SciPy")
    else:  # pragma: no cover
        pytest.skip(f"Unknown solver {name}")
    return name


# ----------------- tests -----------------

def test_linear_kernel_matches_sklearn(solver_name):
    X, y = _linear_toy(n_per=30, sep=2.0, noise=0.1, seed=1)
    C = 1.0

    k = _make_kernel("linear", {})
    ours = SVM(C=C, kernel=k, solver=solver_name, tol=1e-6, verbose=False).fit(X, y)
    sk   = SVC(C=C, kernel="linear").fit(X, y)

    # Accuracy parity on train
    acc_ours = ours.score(X, y)
    acc_sk   = sk.score(X, y)
    assert abs(acc_ours - acc_sk) <= 1e-3

    # Decision sign parity (allow a global sign flip)
    df_ours = np.sign(ours.decision_function(X))
    df_sk   = np.sign(sk.decision_function(X))
    sign_match    = np.mean(df_ours == df_sk)
    sign_mismatch = np.mean(df_ours == -df_sk)
    assert max(sign_match, sign_mismatch) > 0.98

    # Compare w & b (up to sign)
    assert ours.coef_ is not None
    w_ours = ours.coef_.ravel()
    b_ours = float(ours.intercept_)
    w_sk   = sk.coef_.ravel()
    b_sk   = float(sk.intercept_[0])

    err_same = np.linalg.norm(w_ours - w_sk) + abs(b_ours - b_sk)
    err_flip = np.linalg.norm(w_ours + w_sk) + abs(b_ours + b_sk)
    assert min(err_same, err_flip) < 5e-2

    # Basics
    assert ours.support_vectors_.shape[0] >= 1
    assert ours.alpha_ is not None and ours.alpha_.shape[0] == X.shape[0]
    scores = ours.decision_function(X)
    assert scores.shape == (X.shape[0],)


@pytest.mark.parametrize(
    "kernel_name,params",
    [
        ("rbf",  dict(C=1.0, gamma=0.7)),
        ("poly", dict(C=1.0, degree=3, gamma=0.5, coef0=1.0)),
        # You can add ("sigmoid", dict(C=1.0, gamma=0.7, coef0=0.0)) if desired
    ],
)
def test_kernels_moons_fast(kernel_name, params, solver_name):
    X, y = _moons(n=120, noise=0.2, seed=2)

    k = _make_kernel(kernel_name, params)
    ours = SVM(kernel=k, solver=solver_name, tol=1e-6, verbose=False, C=params.get("C", 1.0)).fit(X, y)

    sk = SVC(**_mirror_sk_params(kernel_name, params)).fit(X, y)

    # acc_ours = ours.score(X, y)
    # acc_sk   = sk.score(X, y)
    
    yhat_ours = ours.predict(X)
    yhat_sk   = sk.predict(X)

    acc_ours = mcc(y, yhat_ours)
    acc_sk   = mcc(y, yhat_sk)
    print(f"Kernel {kernel_name}: acc_ours={acc_ours:.4f}, acc_sk={acc_sk:.4f}")
    # Both should do well; allow some slack vs sklearn due to different solvers
    assert acc_ours >= 0.9
    assert acc_ours >= acc_sk - 0.06


def test_predict_and_score_interfaces(solver_name):
    X, y = make_blobs(n_samples=80, centers=2, random_state=3, cluster_std=1.1)
    X = _standardize(X)

    k = _make_kernel("rbf", dict(gamma=0.6))
    clf = SVM(C=0.5, kernel=k, solver=solver_name, tol=1e-6).fit(X, y)

    yhat = clf.predict(X)
    assert yhat.shape == y.shape
    assert 0.0 <= clf.score(X, y) <= 1.0


def test_dual_feasibility_basic(solver_name):
    X, y = _linear_toy(n_per=20, sep=1.8, noise=0.15, seed=4)
    C = 1.2
    k = _make_kernel("linear", {})
    clf = SVM(C=C, kernel=k, solver=solver_name, tol=1e-7).fit(X, y)

    alpha = clf.alpha_
    assert alpha is not None
    # Bounds
    assert np.all(alpha >= -1e-8)
    assert np.all(alpha <= C + 1e-8)
    # Equality: y^T alpha = 0 (for mapped {-1,+1} labels)
    y_mapped = clf.y_
    assert y_mapped is not None
    assert abs(float(y_mapped @ alpha)) < 1e-6
    
    
def test_osqp_solver_params_applied(monkeypatch):
    """Ensure OSQP solver options are forwarded correctly."""
    osqp = pytest.importorskip("osqp", reason="OSQP not installed")
    pytest.importorskip("scipy", reason="OSQP requires SciPy")

    # Capture setup/solve kwargs to verify propagation from solver_params
    captured: dict[str, dict[str, Any]] = {}
    original_setup = osqp.OSQP.setup
    original_solve = osqp.OSQP.solve

    def spy_setup(self, *args, **kwargs):  # type: ignore[no-redef]
        captured["setup"] = kwargs
        return original_setup(self, *args, **kwargs)

    def spy_solve(self, *args, **kwargs):  # type: ignore[no-redef]
        captured["solve"] = kwargs
        return original_solve(self, *args, **kwargs)

    monkeypatch.setattr(osqp.OSQP, "setup", spy_setup)
    monkeypatch.setattr(osqp.OSQP, "solve", spy_solve)

    X, y = _linear_toy(n_per=15, sep=2.0, noise=0.1, seed=5)
    params = {"setup": {"max_iter": 1000, "eps_abs": 1e-7}}
    SVM(C=1.0, kernel=Linear(), solver="osqp", solver_params=params).fit(X, y)

    assert captured["setup"]["max_iter"] == 1000
    assert captured["setup"]["eps_abs"] == pytest.approx(1e-7)
    assert captured["solve"]["raise_error"] is False
