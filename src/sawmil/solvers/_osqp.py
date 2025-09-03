# quadprog_osqp.py
from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np
import numpy.typing as npt
from .objective import Objective
import logging

log = logging.getLogger("solvers._osqp")


def quadprog_osqp(
    H: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    Aeq: Optional[npt.NDArray[np.float64]],
    beq: Optional[npt.NDArray[np.float64]],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    verbose: bool = False,
    **params: Any,
    ) -> Tuple[npt.NDArray[np.float64], "Objective"]:
    """
    Solve the quadratic program using OSQP:

        minimize   0.5 * αᵀ H α + fᵀ α
        subject to Aeq α = beq
                   lb ≤ α ≤ ub
                   
    Args:
        H: (n, n) Hessian matrix for the quadratic term in 0.5 * αᵀ H α. For SVM duals, this is typically (y yᵀ) ⊙ K where K is the kernel matrix.
        f: (n,) linear term vector in fᵀ α. For SVMs, usually -1 for each component.
        Aeq: (m, n) optional equality constraint matrix, e.g. yᵀ for SVM bias constraint.
        beq: (m,) optional right-hand side of equality constraint, usually 0.
        lb: (n,) lower bound vector, e.g. all zeros in standard SVM dual.
        ub: (n,) upper bound vector, e.g. all entries equal to C in soft-margin SVM.
        verbose: If True, print solver logs

    Returns:
        α*: Optimal solution vector
        Objective: quadratic and linear parts of the optimum
    """
    # Lazy, guarded imports so the module can be imported without OSQP installed.
    try:
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for solver='osqp'") from exc
    try:
        import osqp
    except Exception as exc:  # pragma: no cover
        raise ImportError("osqp is required for solver='osqp'") from exc


    n = H.shape[0]
    P = sp.csc_matrix(H)
    q = f

    blocks = []
    l_list, u_list = [], []

    # Equalities: encode as Aeq x in [beq, beq]
    if Aeq is not None:
        blocks.append(sp.csc_matrix(Aeq))
        l_list.append(beq)  # type: ignore[arg-type]
        u_list.append(beq)  # type: ignore[arg-type]

    # Bounds: I x in [lb, ub]
    csc_eye = sp.eye(n, format="csc")
    blocks.append(csc_eye)
    l_list.append(lb)
    u_list.append(ub)

    A = sp.vstack(blocks, format="csc")
    lower_vec = np.concatenate(l_list)
    upper_vec = np.concatenate(u_list)

    # Split options into setup/solve; accept flat options as setup()
    setup_opts: dict[str, Any] = dict(params.get("setup", {}))
    solve_opts: dict[str, Any] = dict(params.get("solve", {}))
    if not setup_opts and not solve_opts:
        # treat all non-nested options as setup()
        setup_opts = {k: v for k, v in params.items() if k not in ("setup", "solve")}

    # sensible defaults; allow override via setup_opts
    setup_defaults = dict(
        polishing=True,   
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=20000,
    )
    for k, v in setup_defaults.items():
        setup_opts.setdefault(k, v)

    # And in solve() default to not raising so we can return diagnostics
    solve_opts.setdefault("raise_error", False)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=lower_vec, u=upper_vec, verbose=verbose, **setup_opts)
    res = prob.solve(**solve_opts)

    if res.info.status_val not in (1, 2):  # 1=solved, 2=solved_inaccurate
        log.error(f"OSQP failed: {res.info.status}")

    x = np.asarray(res.x, dtype=float)
    quadratic = float(0.5 * x @ (H @ x))
    linear = float(f @ x)
    return x, Objective(quadratic, linear)
