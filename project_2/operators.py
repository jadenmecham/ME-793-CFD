import numpy as np

# gradient operator
def G(p, delta, ip, iu, iv, nx, ny, n_vel):
    grad = np.zeros(n_vel)
    P = p[ip]   # reshape flat pressure vector to (nx, ny) via ip pointer array

    # u-faces: between pressure cells (i-1, j) and (i, j), for i = 1..nx-1
    grad[iu[1:nx, :ny]] = (P[1:nx, :] - P[:nx-1, :]) / delta

    # v-faces: between pressure cells (i, j-1) and (i, j), for j = 1..ny-1
    grad[iv[:nx, 1:ny]] = (P[:, 1:ny] - P[:, :ny-1]) / delta

    return grad

# divergence operator
def D():
    pass

# laplacian operator
def L():
    pass

# boundary condition vecotrs
def bc_D():
    pass

def bc_L():
    pass

# advection operator
def A():
    pass

# conjugate gradient solver
def CG():
    pass

