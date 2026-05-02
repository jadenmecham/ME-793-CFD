import numpy as np
import matplotlib.pyplot as plt


def disk_force(C_T, iu, delta, nx, ny, n_vel, xL, yB, x_disk, y_lo, y_hi):
    """
    Body-force vector for a thin actuator disk oriented in the y-direction.

    u-face iu[i,j] sits at x = xL + i*delta, y = yB + (j+0.5)*delta.
    Force per unit volume (non-dim, rho=v_h=D=1):
        f = C_T / (2 * delta)
    Integrating over the disk area (delta * D) gives total thrust T = C_T/2.

    Froude-Rankine ideal hover: C_T = 4 (v_h at disk, 2*v_h in far wake).
    """
    f = np.zeros(n_vel)
    i_disk = int(round((x_disk - xL) / delta))
    i_disk = int(np.clip(i_disk, 1, nx - 1))
    # y-coord of u-face j: yB + (j + 0.5)*delta
    j_lo = max(0,    int(np.floor((y_lo - yB) / delta - 0.5)) + 1)
    j_hi = min(ny - 1, int(np.floor((y_hi - yB) / delta - 0.5)))
    f[iu[i_disk, j_lo:j_hi + 1]] = C_T / (2.0 * delta)
    return f, i_disk, j_lo, j_hi


def check_thrust(f, delta):
    """Integrate body force over the domain; should equal prescribed T = C_T/2."""
    return f.sum() * delta ** 2


def extract_fields(U, P, ip, iu, iv, nx, ny):
    """Reconstruct 2D arrays from flat solution vectors."""
    u_2d = U[iu[1:nx, :ny]]      # (nx-1, ny)
    v_2d = U[iv[:nx, 1:ny]]      # (nx,   ny-1)
    p_2d = P[ip]                  # (nx,   ny)
    return u_2d, v_2d, p_2d


def plot_centerline(xu, xv, yp, u_2d, v_2d, x_disk, j_center, i_disk, j_lo, j_hi, Re):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Streamwise centerline: u along x at y = disk center
    axes[0].plot(xu, u_2d[:, j_center], 'b-', label='CFD')
    axes[0].axvline(x_disk, color='k', linestyle='--', linewidth=1, label='disk')
    axes[0].axhline(1.0, color='r', linestyle=':', linewidth=1, label=r'$v_h$ theory')
    axes[0].axhline(2.0, color='g', linestyle=':', linewidth=1, label=r'$2v_h$ theory')
    axes[0].set_xlabel('x / D')
    axes[0].set_ylabel('u / v_h')
    axes[0].set_title(f'Streamwise centerline velocity  Re={Re}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Disk-plane profile: u vs y at x = x_disk
    y_u = yp  # u-faces share y-coords with pressure centers
    axes[1].plot(u_2d[i_disk - 1, :], y_u, 'b-', label='CFD (disk plane)')
    axes[1].axvline(1.0, color='r', linestyle=':', linewidth=1, label=r'$v_h$ theory')
    axes[1].axhspan(yp[j_lo], yp[j_hi], alpha=0.15, color='orange', label='disk span')
    axes[1].set_xlabel('u / v_h')
    axes[1].set_ylabel('y / D')
    axes[1].set_title(f'Disk-plane velocity profile  Re={Re}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sensor_placement(xp, yp, u_2d, v_2d, xu, yv, x_disk, y_lo, y_hi, Re,
                           threshold=0.05):
    """
    Contour |u| (interpolated to cell centers) and shade regions below threshold
    as candidate sensor locations.
    """
    # Interpolate u to cell centers: average left and right u-faces
    u_cc = 0.5 * (u_2d[:-1, :] + u_2d[1:, :])   # (nx-2, ny) — one fewer in x
    xcc = 0.5 * (xu[:-1] + xu[1:])

    speed = np.abs(u_cc)

    fig, ax = plt.subplots(figsize=(10, 5))
    cf = ax.contourf(xcc, yp, speed.T, levels=50, cmap='viridis')
    plt.colorbar(cf, ax=ax, label=r'$|u|/v_h$')
    cs = ax.contour(xcc, yp, speed.T, levels=[threshold], colors='red', linewidths=2)
    ax.clabel(cs, fmt=lambda x: f'{threshold:.0%} threshold')
    ax.axvline(x_disk, color='white', linestyle='--', linewidth=1.5, label='disk')
    ax.axhspan(y_lo, y_hi, alpha=0.3, color='cyan', label='disk span')
    ax.set_xlabel('x / D')
    ax.set_ylabel('y / D')
    ax.set_title(f'Sensor placement map  Re={Re}  (red = |u| < {threshold:.0%} v_h)')
    ax.legend(loc='upper right')
    return fig


def plot_streamlines(xp, yp, xu, yv, u_2d, v_2d, x_disk, y_lo, y_hi, Re):
    """Streamline plot with velocity magnitude background."""
    # Interpolate u and v to a common cell-center grid for streamplot
    u_cc = np.zeros((len(xp), len(yp)))
    v_cc = np.zeros((len(xp), len(yp)))

    nx = len(xp)
    ny = len(yp)
    u_cc[1:-1, :] = 0.5 * (u_2d[:-1, :] + u_2d[1:, :])
    u_cc[0,    :] = u_2d[0, :]
    u_cc[-1,   :] = u_2d[-1, :]

    v_cc[:, 1:-1] = 0.5 * (v_2d[:, :-1] + v_2d[:, 1:])
    v_cc[:, 0   ] = v_2d[:, 0]
    v_cc[:, -1  ] = v_2d[:, -1]

    speed = np.sqrt(u_cc ** 2 + v_cc ** 2)

    fig, ax = plt.subplots(figsize=(12, 5))
    cf = ax.contourf(xp, yp, speed.T, levels=50, cmap='Blues')
    plt.colorbar(cf, ax=ax, label=r'$|\mathbf{u}|/v_h$')
    ax.streamplot(xp, yp, u_cc.T, v_cc.T, color='k', density=2, linewidth=0.6,
                  arrowsize=0.8)
    ax.axvline(x_disk, color='red', linestyle='--', linewidth=2, label='disk')
    ax.axhspan(y_lo, y_hi, alpha=0.2, color='orange', label='disk span')
    ax.set_xlabel('x / D')
    ax.set_ylabel('y / D')
    ax.set_title(f'Streamlines  Re={Re}')
    ax.legend()
    return fig
