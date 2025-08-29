from __future__ import annotations
import numpy as np
from numpy import typing as npt
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

log = logging.getLogger("quadprog")

@dataclass
class Objective:
    """Container for quadratic/linear objective parts."""

    quadratic: float
    linear: float

    @property
    def objective(self) -> float:
        return self.quadratic + self.linear

def quadprog_gurobi(
    H: npt.NDArray[np.float64],     
    f: npt.NDArray[np.float64],       
    Aeq: Optional[npt.NDArray[np.float64]], 
    beq: Optional[npt.NDArray[np.float64]],  
    lb: npt.NDArray[np.float64],     
    ub: npt.NDArray[np.float64],      
    verbose: bool = False,           
)-> Tuple[npt.NDArray[np.float64], "Objective"]:
    """
    Solve the quadratic program:

        minimize   0.5 * αᵀ H α + fᵀ α
        subject to Aeq α = beq
                   lb ≤ α ≤ ub

    Args:
        H: (n, n) quadratic term matrix in 0.5 * αᵀ H α
        f: (n,) linear term vector in fᵀ α , usually f = -1
        Aeq: (m, n) equality constraint matrix, usually yᵀ
        beq: (m,) equality constraint rhs, usually 0
        lb: (n,) lower bound vector, usually 0
        ub: (n,) upper bound vector, usually C
        verbose: If True, print solver logs

    Returns:
        α*: Optimal solution vector
        Objective: quadratic and linear parts of the optimum
    """

    # ---- VALIDATION
    try:  # pragma: no cover - gurobi may be unavailable in some envs
        import gurobipy as gp
    except Exception as exc:  # pragma: no cover
        raise ImportError("gurobipy is required for solver='gurobi'") from exc
    if (Aeq is None) ^ (beq is None):
        raise ValueError("Aeq and beq must both be None or both be provided.")

    n = H.shape[0]
    # ---- COERCION and VALIDATION
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if H.shape != (n, n):
        raise ValueError(f"H must be (n,n); got {H.shape}")
    if f.shape != (n,):
        raise ValueError(f"f must be (n,); got {f.shape}")
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError("lb and ub must be length-n vectors.")
    if Aeq is not None:
        Aeq = np.asarray(Aeq, dtype=float)
        beq = np.asarray(beq, dtype=float)
        if Aeq.shape[1] != n:
            raise ValueError(f"Aeq must have n columns; got {Aeq.shape}")
        if beq.shape != (Aeq.shape[0],):
            raise ValueError(f"beq must match Aeq rows; got beq {beq.shape}, Aeq {Aeq.shape}")

    # ---- STABILITY
    H = 0.5 * (H + H.T)

    # --- Optimization
    model = gp.Model()
    if not verbose:
        model.Params.OutputFlag = 0
        
    # should we include? 
    # model.Params.NumericFocus = 1
    # model.Params.OptimalityTol = 1e-8
    # model.Params.BarConvTol = 1e-10

    x = model.addMVar(n, lb=lb, ub=ub, name="alpha")
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