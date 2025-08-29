# quadprog_osqp.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from .objective import Objective
import logging

log = logging.getLogger("quadprog_osqp")


def quadprog_osqp(
    H: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    Aeq: Optional[npt.NDArray[np.float64]],
    beq: Optional[npt.NDArray[np.float64]],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    verbose: bool = False,
    ridge: float = 0.0,  # add for stability if needed
) -> Tuple[npt.NDArray[np.float64], "Objective"]:
    """Solve min 0.5 x^T H x + f^T x  s.t.  Aeq x = beq,  lb <= x <= ub using OSQP."""
    # Lazy, guarded imports so the module can be imported without OSQP installed.
    try:
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for solver='osqp'") from exc
    try:
        import osqp
    except Exception as exc:  # pragma: no cover
        raise ImportError("osqp is required for solver='osqp'") from exc

    # ---- shape & sanity checks ----
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    n = H.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"H must be (n,n); got {H.shape}")
    if f.shape != (n,):
        raise ValueError(f"f must be shape (n,); got {f.shape}")

    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError("lb and ub must be shape (n,)")
    if np.any(lb > ub):
        raise ValueError("Each component must satisfy lb[i] <= ub[i].")

    if (Aeq is None) ^ (beq is None):
        raise ValueError("Provide both Aeq and beq, or neither.")
    if Aeq is not None:
        Aeq = np.atleast_2d(np.asarray(Aeq, dtype=float))
        beq = np.asarray(beq, dtype=float).ravel()
        if Aeq.shape[1] != n:
            raise ValueError(f"Aeq must have {n} columns; got {Aeq.shape[1]}")
        if beq.shape != (Aeq.shape[0],):
            raise ValueError("beq shape must match number of Aeq rows.")

    # ---- OSQP data ----
    # Symmetrize and (optionally) add tiny ridge for numerical stability
    P = 0.5 * (H + H.T)
    if ridge > 0.0:
        P = P + ridge * np.eye(n)
    P = sp.csc_matrix(P)
    q = f

    blocks = []
    l_list, u_list = [], []

    # Equalities: encode as Aeq x in [beq, beq]
    if Aeq is not None:
        blocks.append(sp.csc_matrix(Aeq))
        l_list.append(beq)  # type: ignore[arg-type]
        u_list.append(beq)  # type: ignore[arg-type]

    # Bounds: I x in [lb, ub]
    I = sp.eye(n, format="csc")
    blocks.append(I)
    l_list.append(lb)
    u_list.append(ub)

    A = sp.vstack(blocks, format="csc")
    l = np.concatenate(l_list)
    u = np.concatenate(u_list)

    # ---- Solve ----
    prob = osqp.OSQP()
    prob.setup(
        P=P, q=q, A=A, l=l, u=u,
        verbose=verbose,
        polishing=True,
        eps_abs=1e-8, eps_rel=1e-8,
        max_iter=20000,
        # alpha=1.6, adaptive_rho=True
    )
    res = prob.solve(raise_error=False)

    if res.info.status_val not in (1, 2):  # 1=solved, 2=solved_inaccurate
        log.error(f"OSQP failed: {res.info.status}")

    x = np.asarray(res.x, dtype=float)
    quadratic = float(0.5 * x @ (H @ x))
    linear = float(f @ x)
    return x, Objective(quadratic, linear)
