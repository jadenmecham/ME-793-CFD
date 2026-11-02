"""
Regenerate the README animations for Projects 2 and 3.

Re-runs the fractional-step solvers from project_2.ipynb (lid-driven cavity)
and project_3.ipynb (actuator disk) with the same discretization, recording
snapshots of the velocity field, and writes them out as GIFs.

Usage (from the repo root):
    python scripts/make_animations.py            # both
    python scripts/make_animations.py cavity     # Project 2 only
    python scripts/make_animations.py disk       # Project 3 only
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse.linalg import cg, LinearOperator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'project_2'))
sys.path.insert(0, os.path.join(ROOT, 'project_3'))
import operators as ops      # noqa: E402
import actuator_disk as ad   # noqa: E402

FIGSIZE = (6, 5)   # matches project_1/solution_animation.gif
DPI = 100


# ─────────────────────────────────────────────────────────────────────────────
# Shared MAC-grid solver
# ─────────────────────────────────────────────────────────────────────────────
def gen_pointer(nx, ny):
    ip = np.arange(nx * ny).reshape(nx, ny)
    iu = np.full((nx + 1, ny), -1, dtype=int)
    iu[1:nx, :] = np.arange((nx - 1) * ny).reshape(nx - 1, ny)
    iv = np.full((nx, ny + 1), -1, dtype=int)
    iv[:, 1:ny] = (ny * (nx - 1) + np.arange(nx * (ny - 1))).reshape(nx, ny - 1)
    return ip, iu, iv


def run(nx, ny, delta, dt, visc, n_steps, bcs, every, body_force=None):
    """Fractional-step solver (AB2 advection, CN diffusion). Returns snapshots."""
    ip, iu, iv = gen_pointer(nx, ny)
    n_cells = nx * ny
    n_vel = ny * (nx - 1) + nx * (ny - 1)
    uL, uR, uB, uT, vL, vR, vB, vT = bcs
    f = np.zeros(n_vel) if body_force is None else body_force

    lap = lambda x: ops.L(x, delta, iu, iv, nx, ny, n_vel)
    oD = lambda x: ops.D(x, delta, ip, iu, iv, nx, ny, n_cells)
    oG = lambda x: ops.G(x, delta, ip, iu, iv, nx, ny, n_vel)
    oS = lambda x: x + (dt / 2.0 * visc) * lap(x)
    oR = lambda x: x - (dt / 2.0 * visc) * lap(x)

    def oT(x):
        y = oD(oS(oG(x)))
        y[ip[0, 0]] = 0.0
        return y

    def solve(op, b):
        A = LinearOperator((len(b), len(b)), matvec=op, dtype=np.float64)
        return cg(A, b, rtol=1e-8)[0]

    bcL = ops.bc_L(uL, uR, uB, uT, vL, vR, vB, vT, delta, iu, iv, nx, ny, n_vel)
    bcD = ops.bc_D(uL, uR, vB, vT, delta, ip, nx, ny, n_cells)

    U = np.zeros(n_vel)
    adv_prev = np.zeros(n_vel)
    snaps = [(0.0, U[iu[1:nx, :]].copy(), U[iv[:, 1:ny]].copy())]
    for step in range(1, n_steps + 1):
        a0 = ops.A(U, uL, uR, uB, uT, vL, vR, vB, vT, delta, iu, iv, nx, ny, n_vel)
        RHS_U = oS(U) + dt * (1.5 * a0 - 0.5 * adv_prev) + dt * visc * bcL + dt * f
        UF = solve(oR, RHS_U)
        RHS_P = oD(UF) / dt + bcD / dt
        RHS_P[ip[0, 0]] = 0.0
        P = solve(oT, RHS_P)
        U = UF - dt * oS(oG(P))
        adv_prev = a0
        if step % every == 0:
            snaps.append((step * dt, U[iu[1:nx, :]].copy(), U[iv[:, 1:ny]].copy()))
            print(f'\r  step {step:6d}/{n_steps}', end='', flush=True)
    print()
    return snaps, iu


def to_cell_centers(u2d, v2d, nx, ny, uL, uR, vB, vT):
    """Average face velocities to cell centers (including boundary faces)."""
    u_ext = np.zeros((nx + 1, ny))
    u_ext[0, :], u_ext[-1, :], u_ext[1:nx, :] = uL, uR, u2d
    v_ext = np.zeros((nx, ny + 1))
    v_ext[:, 0], v_ext[:, -1], v_ext[:, 1:ny] = vB, vT, v2d
    return 0.5 * (u_ext[:-1] + u_ext[1:]), 0.5 * (v_ext[:, :-1] + v_ext[:, 1:])


def save_gif(fig, update, n_frames, path, fps=12):
    update(n_frames - 1)
    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=DPI)
    plt.close(fig)
    print(f'  saved {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1e6:.1f} MB)')


# ─────────────────────────────────────────────────────────────────────────────
# Project 2 — lid-driven cavity
# ─────────────────────────────────────────────────────────────────────────────
def cavity(Re=1000, nx=65, t_end=10000 * 0.1 / 65, n_frames=100):
    print(f'Project 2: lid-driven cavity, Re={Re}, {nx}x{nx}')
    ny = nx
    delta = 1.0 / nx
    dt = 0.1 * delta
    n_steps = int(round(t_end / dt))
    every = max(1, n_steps // n_frames)
    bcs = (np.zeros(ny), np.zeros(ny), np.zeros(nx - 1), np.ones(nx - 1),
           np.zeros(ny - 1), np.zeros(ny - 1), np.zeros(nx), np.zeros(nx))
    snaps, _ = run(nx, ny, delta, dt, 1.0 / Re, n_steps, bcs, every)

    xp = (np.arange(nx) + 0.5) * delta
    fields = [to_cell_centers(u, v, nx, ny, 0.0, 0.0, 0.0, 0.0) for _, u, v in snaps]
    levels = np.linspace(0, 1.0, 21)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label=r'$|\mathbf{u}|\,/\,U_{lid}$')

    def update(k):
        ax.cla()
        t = snaps[k][0]
        uc, vc = fields[k]
        speed = np.sqrt(uc ** 2 + vc ** 2)
        ax.contourf(xp, xp, speed.T, levels=levels, cmap='magma', extend='max')
        if k > 0:
            ax.streamplot(xp, xp, uc.T, vc.T, density=1.4, color='w',
                          linewidth=0.5, arrowsize=0.6)
        ax.annotate('', xy=(0.85, 1.025), xytext=(0.15, 1.025), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.5), annotation_clip=False)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'$x$', ylabel=r'$y$', aspect='equal')
        ax.set_title(f'Lid-driven cavity, Re = {Re}    t = {t:5.1f}', fontsize=11, pad=14)

    out = os.path.join(ROOT, 'project_2', 'cavity_animation.gif')
    save_gif(fig, update, len(snaps), out)


# ─────────────────────────────────────────────────────────────────────────────
# Project 3 — actuator disk
# ─────────────────────────────────────────────────────────────────────────────
def disk(Re=200, ny=100, t_end=10.0, n_frames=100, C_T=4.0):
    print(f'Project 3: actuator disk, Re={Re}, C_T={C_T}, {ny}x{ny}')
    L = 10.0
    delta = L / ny
    nx = ny
    n_vel = ny * (nx - 1) + nx * (ny - 1)
    dt = min(0.5 * delta / 2.0, delta ** 2 * Re / 2.0)
    n_steps = int(round(t_end / dt))
    every = max(1, n_steps // n_frames)
    _, iu, _ = gen_pointer(nx, ny)
    x_disk, y_lo, y_hi = 5.0, 4.5, 5.5
    f, *_ = ad.disk_force(C_T, iu, delta, nx, ny, n_vel, 0.0, 0.0, x_disk, y_lo, y_hi)
    bcs = (np.zeros(ny), np.zeros(ny), np.zeros(nx - 1), np.zeros(nx - 1),
           np.zeros(ny - 1), np.zeros(ny - 1), np.zeros(nx), np.zeros(nx))
    snaps, _ = run(nx, ny, delta, dt, 1.0 / Re, n_steps, bcs, every, body_force=f)

    xp = (np.arange(nx) + 0.5) * delta
    fields = [to_cell_centers(u, v, nx, ny, 0.0, 0.0, 0.0, 0.0) for _, u, v in snaps]
    vmax = 2.0
    levels = np.linspace(0, vmax, 21)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, label=r'$|\mathbf{u}|\,/\,v_h$')

    def update(k):
        ax.cla()
        t = snaps[k][0]
        uc, vc = fields[k]
        speed = np.sqrt(uc ** 2 + vc ** 2)
        ax.contourf(xp, xp, speed.T, levels=levels, cmap='viridis', extend='max')
        if k > 0:
            ax.streamplot(xp, xp, uc.T, vc.T, density=1.2, color='w',
                          linewidth=0.5, arrowsize=0.6)
        ax.plot([x_disk, x_disk], [y_lo, y_hi], color='r', lw=3, solid_capstyle='butt')
        ax.set(xlim=(0, L), ylim=(0, L), xlabel=r'$x\,/\,D$', ylabel=r'$y\,/\,D$', aspect='equal')
        ax.set_title(f'Actuator disk, Re = {Re}, $C_T$ = {C_T:g}    t = {t:4.1f}', fontsize=11)

    out = os.path.join(ROOT, 'project_3', 'actuator_disk_animation.gif')
    save_gif(fig, update, len(snaps), out)


if __name__ == '__main__':
    which = sys.argv[1:] or ['cavity', 'disk']
    if 'cavity' in which:
        cavity()
    if 'disk' in which:
        disk()
