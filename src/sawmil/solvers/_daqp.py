# solvers/_daqp.py
import numpy as np
import numpy.typing as npt

def quadprog_daqp(H, f, Aeq, beq, lb, ub, verbose=False):
    import scipy.sparse as sp
    import daqp

    n = H.shape[0]
    # DAQP form: min ½xᵀPx + qᵀx  s.t. Ax ≤ b, l ≤ x ≤ u
    # Put the equality as two inequalities: Aeq x ≤ beq and -Aeq x ≤ -beq
    P = H
    q = f.copy()

    A_list, b_list = [], []
    if Aeq is not None:
        Aeq = np.atleast_2d(Aeq)
        beq = np.asarray(beq).ravel()
        A_list += [ Aeq, -Aeq ]
        b_list += [ beq,  -beq ]
    A = sp.vstack([sp.csr_matrix(A) for A in A_list]) if A_list else sp.csr_matrix((0,n))
    b = np.concatenate(b_list) if b_list else np.zeros((0,))

    settings = daqp.Settings()
    settings.verbose = int(verbose)

    sol = daqp.solve_qp(P, q, A, b, lb, ub, settings=settings)
    if sol["flag"] not in (1,):  # 1 = optimal
        raise RuntimeError(f"DAQP failed with flag {sol['flag']}")
    x = np.asarray(sol["x"]).ravel()
    # objective (match your Objective class if you have one)
    obj_quad = float(0.5 * x @ (H @ x))
    obj_lin  = float(f @ x)
    return x, (obj_quad, obj_lin)
