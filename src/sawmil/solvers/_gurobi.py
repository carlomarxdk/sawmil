from __future__ import annotations
import numpy as np
from numpy import typing as npt
from typing import Tuple, Optional, Any, Dict
from .objective import Objective
import logging

log = logging.getLogger("solvers.gurobi")

def quadprog_gurobi(
    H: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    Aeq: Optional[npt.NDArray[np.float64]],
    beq: Optional[npt.NDArray[np.float64]],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    verbose: bool = False,
    **params: Optional[Dict],

) -> Tuple[npt.NDArray[np.float64], "Objective"]:
    """
    Solve the quadratic program using Gurobi:

        minimize   0.5 * αᵀ H α + fᵀ α
        subject to Aeq α = beq
                   lb ≤ α ≤ ub

    Args:
        H (np.ndarray): (n, n) Hessian matrix for the quadratic term in 0.5 * αᵀ H α.
        f (np.ndarray): (n,) linear term vector in fᵀ α. For SVMs, usually -1 for each component.
        Aeq (np.ndarray | None): (m, n) equality constraint matrix, usually yᵀ. Must be provided iff ``beq`` is provided.
        beq (np.ndarray | None): (m,) equality constraint right-hand side, usually 0. Must be provided iff ``Aeq`` is provided.
        lb (np.ndarray): (n,) lower bound vector, usually 0.
        ub (np.ndarray): (n,) upper bound vector, usually C.
        verbose (bool): If True, print solver logs.
        params (Any): Additional keyword parameters passed via ``**params``.
            Expected keys:
            * ``env`` (dict): Parameters for ``gurobipy.Env`` (e.g., ``{"LogFile": "gurobi.log"}``).
            * ``model`` (dict): Parameters for ``gurobipy.Model.Params`` (e.g., ``{"Method": 2, "Threads": 1}``).
            * ``start`` (np.ndarray | None): Optional initial solution vector of shape (n,) for warm start.

    Returns:
        x (np.ndarray): Optimal solution vector α* of shape (n,).
        objective (Objective): Quadratic and linear parts of the optimum.

    Raises:
        ImportError: If ``gurobipy`` is not installed.
        ValueError: If only one of ``Aeq`` or ``beq`` is provided, or if a warm start has the wrong shape.
    """

    try:
        import gurobipy as gp
    except Exception as exc:  # pragma: no cover
        raise ImportError("gurobipy is required for solver='gurobi'") from exc
    if (Aeq is None) ^ (beq is None):
        raise ValueError("Aeq and beq must both be None or both be provided.")
    
    n = H.shape[0]
    env_cfg = dict(params.get("env", {}))
    env = gp.Env(**env_cfg) if env_cfg else None
    if env_cfg:
        log.debug(f"Gurobi Env params: {env_cfg}")
    
    model = gp.Model(env=env) if env else gp.Model()
    if not verbose:
        model.Params.OutputFlag = 0

    # Collect model parameter overrides
    model_params: dict[str, Any] = dict(params.get("model", {}))
    # Also allow flat, non-namespaced params as a convenience
    for k, v in params.items():
        if k not in ("model", "env", "start"):
            model_params.setdefault(k, v)
            
    if model_params:
        log.debug(f"Gurobi Model.Params overrides: {model_params}")

    # Apply model.Params.* safely
    for k, v in model_params.items():
        try:
            setattr(model.Params, k, v)
        except AttributeError as exc:
            raise ValueError(f"Unknown Gurobi parameter: '{k}'") from exc


    x = model.addMVar(n, lb=lb, ub=ub, name="alpha")

    if "start" in params and params["start"] is not None:
        start = np.asarray(params["start"], dtype=float)
        if start.shape != (n,):
            raise ValueError(f"start must have shape ({n},), got {start.shape}")
        x.Start = start
    obj = 0.5 * (x @ H @ x) + f @ x
    model.setObjective(obj, gp.GRB.MINIMIZE)

    if Aeq is not None:
        model.addConstr(Aeq @ x == beq, name="eq")
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:  # pragma: no cover - defensive
        log.warning(RuntimeError(
            f"Gurobi optimization failed with status {model.Status}"))

    xstar = np.asarray(x.X, dtype=float)
    quadratic = float(0.5 * xstar.T @ H @ xstar)
    linear = float(f.T @ xstar)
    return xstar, Objective(quadratic, linear)
