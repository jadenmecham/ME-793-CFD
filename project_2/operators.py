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
def D(vel, delta, ip, iu, iv, nx, ny, n_cells):
    div_2d = np.zeros((nx, ny))

    # u-faces: iu[1:nx, :] — east face of cell (i-1,j), west face of cell (i,j)
    u_vals = vel[iu[1:nx, :ny]]           # shape (nx-1, ny)
    div_2d[:nx-1, :] += u_vals / delta    # east → cell (i,j), i=0..nx-2
    div_2d[1:nx,  :] -= u_vals / delta    # west → cell (i,j), i=1..nx-1

    # v-faces: iv[:, 1:ny] — north face of cell (i,j-1), south face of cell (i,j)
    v_vals = vel[iv[:nx, 1:ny]]           # shape (nx, ny-1)
    div_2d[:, :ny-1] += v_vals / delta    # north → cell (i,j), j=0..ny-2
    div_2d[:, 1:ny]  -= v_vals / delta    # south → cell (i,j), j=1..ny-1

    div = np.zeros(n_cells)
    div[ip] = div_2d
    return div

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

