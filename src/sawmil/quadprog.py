from __future__ import annotations
import logging
from typing import Optional, Tuple, Union
from .solvers.objective import Objective

try:
    from .solvers._gurobi import quadprog_gurobi as _qp_gurobi
except Exception:  # ImportError, license errors, etc.
    _qp_gurobi = None
# try:
from .solvers._osqp import quadprog_osqp as _qp_osqp
# except Exception:
#     _qp_osqp = None

import numpy as np
import numpy.typing as npt

log = logging.getLogger("quadprog")

def quadprog(
    H: npt.NDArray[np.float64], 
    f: npt.NDArray[np.float64], 
    Aeq: Optional[npt.NDArray[np.float64]],
    beq: Optional[npt.NDArray[np.float64]], 
    lb: npt.NDArray[np.float64], 
    ub: npt.NDArray[np.float64],
    solver: str = "gurobi", 
    verbose: bool = False
) -> Union[Tuple[npt.NDArray[np.float64], "Objective"], None]:
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
    
    if solver == "gurobi":
        if _qp_gurobi is None:
            raise ImportError("gurobi path selected but 'gurobipy' is not installed or usable. "
                              "Install with: pip install sawmil[gurobi]")
        return _qp_gurobi(H, f, Aeq, beq, lb, ub, verbose=verbose)

    if solver == "osqp":
        # if _qp_osqp is None:
        #     raise ImportError("osqp path selected but 'osqp' is not installed. "
        #                       "Install with: pip install sawmil[osqp]")
        return _qp_osqp(H, f, Aeq, beq, lb, ub, verbose=verbose)
    # if solver == "cvxopt":
        # return quadprog_cvxopt(H, f, Aeq, beq, lb, ub, verbose=verbose)
    raise ValueError(f"Unknown solver: {solver}")
