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
def L(vel, delta, iu, iv, nx, ny, n_vel):
    """
    Homogeneous Laplacian operator (u-type → u-type).

    L_full u = L u + bc_L, where L ignores BC values and bc_L
    carries the Dirichlet wall contributions.

    Boundary stencil rules (Table 1 of Notes_AllOperators):
      u-nodes, left/right walls  : drop the missing x-neighbor (coeff 0)
      u-nodes, bottom/top walls  : ghost = -interior → extra -1/δ² on center
      v-nodes, bottom/top walls  : drop the missing y-neighbor (coeff 0)
      v-nodes, left/right walls  : ghost = -interior → extra -1/δ² on center
    """
    result = np.zeros(n_vel)

    # --- u-nodes: shape (nx-1, ny) ---
    U = vel[iu[1:nx, :ny]]
    Lu = np.zeros_like(U)

    # x-direction (drop missing neighbors at left/right walls)
    Lu -= 2 * U / delta**2
    Lu[:nx-2, :] += U[1:,    :] / delta**2   # right neighbor
    Lu[1:,    :] += U[:nx-2, :] / delta**2   # left neighbor

    # y-direction (ghost = -interior at bottom/top)
    Lu -= 2 * U / delta**2
    Lu[:, 0]    -= U[:, 0]    / delta**2     # extra -1 at bottom
    Lu[:, ny-1] -= U[:, ny-1] / delta**2     # extra -1 at top
    Lu[:, :ny-1] += U[:, 1:]   / delta**2   # above neighbor (j+1)
    Lu[:, 1:]    += U[:, :ny-1] / delta**2  # below neighbor (j-1)

    result[iu[1:nx, :ny]] = Lu

    # --- v-nodes: shape (nx, ny-1) ---
    V = vel[iv[:nx, 1:ny]]
    Lv = np.zeros_like(V)

    # y-direction (drop missing neighbors at bottom/top walls)
    Lv -= 2 * V / delta**2
    Lv[:, :ny-2] += V[:, 1:]   / delta**2   # above neighbor (j+1)
    Lv[:, 1:]    += V[:, :ny-2] / delta**2  # below neighbor (j-1)

    # x-direction (ghost = -interior at left/right)
    Lv -= 2 * V / delta**2
    Lv[0,    :] -= V[0,    :] / delta**2    # extra -1 at left
    Lv[nx-1, :] -= V[nx-1, :] / delta**2   # extra -1 at right
    Lv[:nx-1, :] += V[1:,    :] / delta**2  # right neighbor (i+1)
    Lv[1:,    :] += V[:nx-1, :] / delta**2  # left neighbor (i-1)

    result[iv[:nx, 1:ny]] = Lv

    return result

# boundary condition vectors
def bc_D(uBC_L, uBC_R, vBC_B, vBC_T, delta, ip, nx, ny, n_cells):
    """
    Divergence BC correction vector.

    The homogeneous D skips wall faces (not stored in vel). This vector
    supplies the missing normal-flux contributions at each wall:
      Left  wall: west u-face of cells (0, j)    → -uBC_L / Δx
      Right wall: east u-face of cells (nx-1, j) → +uBC_R / Δx
      Bottom wall: south v-face of cells (i, 0)  → -vBC_B / Δy
      Top   wall: north v-face of cells (i, ny-1)→ +vBC_T / Δy

    BCs can be scalars (uniform wall) or 1-D arrays (per-cell wall).
    """
    bcd_2d = np.zeros((nx, ny))
    bcd_2d[0,    :] -= np.asarray(uBC_L) / delta   # left wall
    bcd_2d[nx-1, :] += np.asarray(uBC_R) / delta   # right wall
    bcd_2d[:,    0] -= np.asarray(vBC_B) / delta   # bottom wall
    bcd_2d[:, ny-1] += np.asarray(vBC_T) / delta   # top wall

    bcd = np.zeros(n_cells)
    bcd[ip] = bcd_2d
    return bcd

def bc_L(uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_B, vBC_T, delta, iu, iv, nx, ny, n_vel):
    """
    Laplacian BC correction vector bc_L.

    Carries the Dirichlet wall contributions omitted by the homogeneous L.
    Factor-of-2 rule (Table 1 of Notes_AllOperators):
      - Node sits AT the wall position (no ghost needed) → factor 1  → BC/δ²
      - Node is half-δ from the wall (ghost-cell treatment)  → factor 2  → 2·BC/δ²

    u-nodes (sit at x-wall positions, need ghost only in y):
      Left  iu[1,   :]  : +uBC_L / δ²      (direct x-BC, factor 1)
      Right iu[nx-1,:]  : +uBC_R / δ²      (direct x-BC, factor 1)
      Bottom iu[1:nx,0] : +2·uBC_B / δ²   (ghost in y, factor 2)
      Top  iu[1:nx,ny-1]: +2·uBC_T / δ²   (ghost in y, factor 2)

    v-nodes (sit at y-wall positions, need ghost only in x):
      Bottom iv[:,1]    : +vBC_B / δ²      (direct y-BC, factor 1)
      Top    iv[:,ny-1] : +vBC_T / δ²      (direct y-BC, factor 1)
      Left   iv[0,1:]   : +2·vBC_L / δ²   (ghost in x, factor 2)
      Right  iv[nx-1,1:]: +2·vBC_R / δ²   (ghost in x, factor 2)
    """
    bcl = np.zeros(n_vel)

    # u-nodes: x-direction BCs (direct, factor 1)
    bcl[iu[1,    :ny]] += np.asarray(uBC_L) / delta**2
    bcl[iu[nx-1, :ny]] += np.asarray(uBC_R) / delta**2

    # u-nodes: y-direction BCs (ghost cell, factor 2)
    bcl[iu[1:nx, 0   ]] += 2 * np.asarray(uBC_B) / delta**2
    bcl[iu[1:nx, ny-1]] += 2 * np.asarray(uBC_T) / delta**2

    # v-nodes: y-direction BCs (direct, factor 1)
    bcl[iv[:nx, 1   ]] += np.asarray(vBC_B) / delta**2
    bcl[iv[:nx, ny-1]] += np.asarray(vBC_T) / delta**2

    # v-nodes: x-direction BCs (ghost cell, factor 2)
    bcl[iv[0,    1:ny]] += 2 * np.asarray(vBC_L) / delta**2
    bcl[iv[nx-1, 1:ny]] += 2 * np.asarray(vBC_R) / delta**2

    return bcl

# advection operator
def A(vel, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_B, vBC_T, delta, iu, iv, nx, ny, n_vel):
    result = np.zeros(n_vel)

    U = vel[iu[1:nx, :ny]]   # (nx-1, ny): u at x=(m+1)δ, y=(j+0.5)δ
    V = vel[iv[:nx, 1:ny]]   # (nx,   ny-1): v at x=(k+0.5)δ, y=(q+1)δ

    # padded arrays for boundary stencils
    Ux = np.empty((nx+1, ny  )); Ux[0, :]     = uBC_L; Ux[1:nx, :]    = U; Ux[nx, :]    = uBC_R
    Uy = np.empty((nx-1, ny+2)); Uy[:, 0]     = uBC_B; Uy[:, 1:ny+1]  = U; Uy[:, ny+1]  = uBC_T
    Vp = np.empty((nx,   ny+1)); Vp[:, 0]     = vBC_B; Vp[:, 1:ny]    = V; Vp[:, ny]    = vBC_T
    Vx = np.empty((nx+2, ny-1)); Vx[0, :]     = vBC_L; Vx[1:nx+1, :]  = V; Vx[nx+1, :]  = vBC_R

    # --- u-component: Ax at each u-node, shape (nx-1, ny) ---
    # x-faces of u-CV: average adjacent u-nodes
    ux_R = 0.5 * (Ux[1:nx,   :      ] + Ux[2:nx+1, :      ])
    ux_L = 0.5 * (Ux[:nx-1,  :      ] + Ux[1:nx,   :      ])
    # y-faces of u-CV: average u-nodes above/below
    uy_T = 0.5 * (Uy[:, 1:ny+1      ] + Uy[:, 2:ny+2      ])
    uy_B = 0.5 * (Uy[:, :ny         ] + Uy[:, 1:ny+1      ])
    # y-faces of u-CV: average v-nodes left/right of face
    vx_T = 0.5 * (Vp[:nx-1, 1:ny+1  ] + Vp[1:nx, 1:ny+1  ])
    vx_B = 0.5 * (Vp[:nx-1, :ny     ] + Vp[1:nx, :ny     ])

    Ax = (ux_R**2 - ux_L**2 + uy_T * vx_T - uy_B * vx_B) / delta
    result[iu[1:nx, :ny]] = Ax

    # --- v-component: Ay at each v-node, shape (nx, ny-1) ---
    # x-faces of v-CV: average adjacent v-nodes
    vx_R_v = 0.5 * (Vx[1:nx+1, :     ] + Vx[2:nx+2, :     ])
    vx_L_v = 0.5 * (Vx[:nx,    :     ] + Vx[1:nx+1, :     ])
    # x-faces of v-CV: average u-nodes above/below face (straddle v-node height)
    uy_R_v = 0.5 * (Ux[1:nx+1, :ny-1 ] + Ux[1:nx+1, 1:ny  ])
    uy_L_v = 0.5 * (Ux[:nx,    :ny-1 ] + Ux[:nx,    1:ny  ])
    # y-faces of v-CV: average adjacent v-nodes
    vy_T_v = 0.5 * (Vp[:, 1:ny       ] + Vp[:, 2:ny+1     ])
    vy_B_v = 0.5 * (Vp[:, :ny-1      ] + Vp[:, 1:ny       ])

    Ay = (uy_R_v * vx_R_v - uy_L_v * vx_L_v + vy_T_v**2 - vy_B_v**2) / delta
    result[iv[:nx, 1:ny]] = Ay

    return result

# conjugate gradient solver
def CG():
    pass

