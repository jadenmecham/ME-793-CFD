import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import spdiags
import matplotlib.patheffects as pe

# -------------------------#
# Laplace matrix functions #
# -------------------------#

# compute the interior points of the laplace operator
def Laplace_u(u, Nx, Ny, index, dx, dy):
    value = np.zeros_like(u)
    # internal nodes
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            idx = index[i, j]
            value[idx] = (u[index[i+1, j]] - 2*u[idx] + u[index[i-1, j]]) / dx**2 + \
                         (u[index[i, j+1]] - 2*u[idx] + u[index[i, j-1]]) / dy**2
    # west boundary (dirichlet)
    i = 0
    for j in range(1, Ny-1):
        idx = index[i, j]
        value[idx] = (-2*u[idx] + u[index[i+1, j]]) / dx**2 + \
                      + (u[index[i, j+1]] - 2*u[idx] + u[index[i, j-1]]) / dy**2
    # east boundary (neumann)
    i = Nx-1
    for j in range(1, Ny-1):
        idx = index[i, j]
        value[idx] = (u[index[i-1, j]] - u[idx]) / dx**2 + \
                      + (u[index[i, j+1]] - 2*u[idx] + u[index[i, j-1]]) / dy**2
    # north boundary (dirichlet)
    j = Ny-1  
    for i in range(1, Nx-1):
        idx = index[i, j]
        value[idx] = (u[index[i+1, j]] - 2*u[idx] + u[index[i-1, j]]) / dx**2 + \
                      + (-2*u[idx] + u[index[i, j-1]]) / dy**2
    # south boundary (neumann)
    j = 0
    for i in range(1, Nx-1):
        idx = index[i, j]
        value[idx] = (u[index[i+1, j]] - 2*u[idx] + u[index[i-1, j]]) / dx**2 + \
                      + (u[index[i, j+1]] - u[idx]) / dy**2
    # corners
    # southwest corner (west dirichlet + south neumann)
    i, j = 0, 0
    idx = index[i, j]
    value[idx] = (-2*u[idx] + u[index[i+1, j]]) / dx**2 + \
                  + (u[index[i, j+1]] - u[idx]) / dy**2
    # southeast corner (east neumann + south neumann)
    i, j = Nx-1, 0
    idx = index[i, j]
    value[idx] = (u[index[i-1, j]] - u[idx]) / dx**2 + \
                  + (u[index[i, j+1]] - u[idx]) / dy**2
    # northwest corner (west dirichlet + north dirichlet)
    i, j = 0, Ny-1
    idx = index[i, j]
    value[idx] = (-2*u[idx] + u[index[i+1, j]]) / dx**2 + \
                  + (-2*u[idx] + u[index[i, j-1]]) / dy**2
    # northeast corner (east neumann + north dirichlet)
    i, j = Nx-1, Ny-1
    idx = index[i, j]
    value[idx] = (u[index[i-1, j]] - u[idx]) / dx**2 + \
                  + (-2*u[idx] + u[index[i, j-1]]) / dy**2
    return value

# deal with the boundary conditions of the laplace operator
def Laplace_BC(Nx, Ny, index, dx, dy, BCL_D, BCR_N, BCB_N, BCT_D):
    value = np.zeros(Nx*Ny)

    # west boundary (dirichlet)
    i = 0
    for j in range(1, Ny-1):
        idx = index[i, j]
        value[idx] = BCL_D[j] / dx**2

    # east boundary (neumann)
    i = Nx-1
    for j in range(1, Ny-1):
        idx = index[i, j]
        value[idx] = BCR_N[j] / dx  

    # north boundary (dirichlet)
    j = Ny-1
    for i in range(1, Nx-1):
        idx = index[i, j]
        value[idx] = BCT_D[i] / dy**2

    # south boundary (neumann)
    j = 0
    for i in range(1, Nx-1):
        idx = index[i, j]
        value[idx] = -BCB_N[i] / dy  

    # corners
    # southwest corner (west dirichlet + south neumann)
    i, j = 0, 0
    idx = index[i, j]
    value[idx] = BCL_D[j] / dx**2 - BCB_N[i] / dy

    # southeast corner (east neumann + south neumann)
    i, j = Nx-1, 0
    idx = index[i, j]
    value[idx] = BCR_N[j] / dx - BCB_N[i] / dy 

    # northwest corner (west dirichlet + north dirichlet)
    i, j = 0, Ny-1
    idx = index[i, j]
    value[idx] = BCL_D[j] / dx**2 + BCT_D[i] / dy**2

    # northeast corner (east neumann + north dirichlet)
    i, j = Nx-1, Ny-1
    idx = index[i, j]
    value[idx] = BCR_N[j] / dx + BCT_D[i] / dy**2 

    return value

# -------------------------#
# Solvers                  #
# -------------------------#

# conjugate gradient method solver
def cg_solver(LHS, RHS, x0, tol, maxit):
    u = x0.copy()
    r = RHS - LHS(u)
    p = r.copy()
    res0 = np.linalg.norm(r)
    resvec = np.zeros(maxit)

    for iter in range(1, maxit + 1):
        LHS_p = LHS(p)
        alpha = (r @ r) / (p @ LHS_p)

        u = u + alpha * p
        r_new = r - alpha * LHS_p

        resvec[iter - 1] = np.linalg.norm(r_new) / res0

        if resvec[iter - 1] < tol:
            resvec = resvec[:iter]
            return u, iter, resvec

        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    resvec = resvec[:iter]
    return u, iter, resvec

# crank-nicolson method solver
def crank_nicolson(Nx, Ny, index, dx, dy, dt, alpha, k_flat, U_ss, 
                   BCL_D, BCR_N, BCB_N, BCT_D, u0=None,
                   maxit_Heat=5000, tol=1e-4, maxit_CG=5000):

    A_CN = lambda u: u - (alpha * dt / 2) * Laplace_u(u, Nx, Ny, index, dx, dy)
    uT = u0.copy() if u0 is not None else np.zeros(Nx * Ny)
    L_BC = Laplace_BC(Nx, Ny, index, dx, dy, BCL_D, BCR_N, BCB_N, BCT_D)
    U_ss_vector = U_ss.reshape(-1, order='F')

    history = [(0.0, uT.reshape(Nx, Ny, order='F').copy())]
    time_stop = None

    for it in range(1, maxit_Heat + 1):
        L_int = Laplace_u(uT, Nx, Ny, index, dx, dy)
        RHS = alpha * dt / 2 * (L_int + 2 * L_BC) + dt * k_flat + uT
        uT, it_cg, res_cg = cg_solver(A_CN, RHS, uT, tol, maxit_CG)

        history.append((it * dt, uT.reshape(Nx, Ny, order='F').copy()))
        print(f'iterations: {it}', end='\r')

        if np.linalg.norm(uT - U_ss_vector, 2) / np.linalg.norm(U_ss_vector, 2) < 1e-4:
            time_stop = it * dt
            print(f'Converged at t = {time_stop:.4f} ({it} timesteps)')
            break

    if time_stop is None:
        print('Did not converge within maximum iterations')

    return history, time_stop

# -------------------------#
# Plotters                 #
# -------------------------#