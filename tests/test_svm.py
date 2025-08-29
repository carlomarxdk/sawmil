# tests/test_svm.py
import numpy as np
import pytest
from sklearn.svm import SVC
from src.sawmil.svm import SVM

from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

# Skips the test if the gurobi is not installed
gp = pytest.importorskip("gurobipy", reason="Gurobi not installed")

@pytest.fixture(scope="session")
def gurobi_env():
    gp = pytest.importorskip("gurobipy", reason="Gurobi not installed")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env

RNG = np.random.default_rng(0)

def _standardize(X):
    return StandardScaler().fit_transform(X)

def _linear_toy(n_per=25, sep=2.0, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(loc=[+sep, +sep], scale=noise, size=(n_per, 2))
    X_neg = rng.normal(loc=[-sep, -sep], scale=noise, size=(n_per, 2))
    X = np.vstack([X_pos, X_neg]).astype(float)
    y = np.hstack([np.ones(n_per), np.zeros(n_per)])  # intentionally not {-1,+1}
    return _standardize(X), y

def _moons(n=120, noise=0.2, seed=2):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return _standardize(X), y

def _mirror_sk_params(kernel, params):
    sk = dict(C=params.get("C", 1.0), kernel=kernel)
    for k in ("gamma", "degree", "coef0"):
        if k in params: sk[k] = params[k]
    return sk

# ---------- Linear kernel: tight parity with sklearn ----------
def test_linear_kernel_matches_sklearn(gurobi_env):
    X, y = _linear_toy(n_per=30, sep=2.0, noise=0.1, seed=1)

    C = 1.0
    ours = SVM(C=C, kernel="linear", tol=1e-6, verbose=False)
    # (Optional) If you wired this through SVM->quadprog, it reuses a single Env
    setattr(ours, "_gurobi_env", gurobi_env)
    setattr(ours, "_gurobi_params", {"Method": 2, "Crossover": 0, "Threads": 1})
    ours.fit(X, y)

    sk = SVC(C=C, kernel="linear").fit(X, y)

    # Accuracy parity
    acc_ours = ours.score(X, y)
    acc_sk   = sk.score(X, y)
    assert abs(acc_ours - acc_sk) <= 1e-3  # tiny slack

    # Decision sign parity (account for possible sign flip)
    df_ours = np.sign(ours.decision_function(X))
    df_sk   = np.sign(sk.decision_function(X))
    sign_match    = np.mean(df_ours == df_sk)
    sign_mismatch = np.mean(df_ours == -df_sk)
    assert max(sign_match, sign_mismatch) > 0.98

    # w & b comparable up to sign
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

# ---------- Fast moons sanity for each non-linear kernel ----------
@pytest.mark.parametrize(
    "kernel,params",
    [
        ("rbf",     dict(C=1.0, gamma=0.7)),
        ("poly",    dict(C=1.0, degree=3, gamma=0.5, coef0=1.0)),
    ],
)
def test_kernels_moons_fast(kernel, params, gurobi_env):
    X, y = _moons(n=120, noise=0.2, seed=2)

    ours = SVM(kernel=kernel, tol=1e-6, verbose=False, **params)
    setattr(ours, "_gurobi_env", gurobi_env)
    setattr(ours, "_gurobi_params", {"Method": 2, "Crossover": 0, "Threads": 1})
    ours.fit(X, y)

    sk = SVC(**_mirror_sk_params(kernel, params)).fit(X, y)

    acc_ours = ours.score(X, y)
    acc_sk   = sk.score(X, y)

    assert acc_ours >= 0.9
    assert acc_ours >= acc_sk - 0.05, f"Expected {acc_ours} >= {acc_sk - 0.05}"

# ---------- Predict/pipeline friendliness ----------
def test_predict_and_score_interfaces(gurobi_env):
    X, y = make_blobs(n_samples=80, centers=2, random_state=3, cluster_std=1.1)
    X = _standardize(X)

    clf = SVM(C=0.5, kernel="rbf", gamma=0.6, tol=1e-6)
    setattr(clf, "_gurobi_env", gurobi_env)
    setattr(clf, "_gurobi_params", {"Method": 2, "Crossover": 0, "Threads": 1})
    clf.fit(X, y)

    yhat = clf.predict(X)
    assert yhat.shape == y.shape
    assert 0.0 <= clf.score(X, y) <= 1.0

# ---------- Basic dual feasibility ----------
def test_dual_feasibility_basic(gurobi_env):
    X, y = _linear_toy(n_per=20, sep=1.8, noise=0.15, seed=4)
    C = 1.2
    clf = SVM(C=C, kernel="linear", tol=1e-7)
    setattr(clf, "_gurobi_env", gurobi_env)
    setattr(clf, "_gurobi_params", {"Method": 2, "Crossover": 0, "Threads": 1})
    clf.fit(X, y)

    alpha = clf.alpha_
    assert alpha is not None
    # Bounds
    assert np.all(alpha >= -1e-8)
    assert np.all(alpha <= C + 1e-8)
    # Equality y^T alpha = 0 in mapped {-1,+1} space
    y_mapped = clf.y_
    assert y_mapped is not None
    assert abs(float(y_mapped @ alpha)) < 1e-6
