# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ME-793 CFD coursework implementing numerical methods for PDEs in Python. Two major projects:

- **Project 1**: 2D heat equation (elliptic + transient) via finite differences and Crank-Nicolson
- **Project 2**: Incompressible Navier-Stokes via the fractional-step (projection) method on a staggered MAC grid

## Environment

A `.venv` (Python 3.8.10) is at the repo root. Activate it before running anything:

```bash
source .venv/bin/activate
```

Key dependencies: `numpy`, `scipy`, `matplotlib`, `jupyter`.

## Running Code

All primary work lives in Jupyter notebooks. To launch:

```bash
jupyter notebook
```

Notebooks use `%autoreload 2` to pick up changes to `utils.py` / `operators.py` without restarting the kernel.

To run a specific notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute project_2/operator_unit_tests.ipynb
```

Unit tests for Project 2 operators live in [project_2/operator_unit_tests.ipynb](project_2/operator_unit_tests.ipynb) — this is the primary way to validate changes to `operators.py`.

## Architecture

### Project 1 — Heat Equation ([project_1/](project_1/))

- [project_1/utils.py](project_1/utils.py) — all numerical kernels:
  - `Laplace_u()` / `Laplace_BC()` — sparse Laplacian matrix and boundary correction vector using mixed BCs (Dirichlet on west/north walls, Neumann on east/south walls)
  - `cg_solver()` — conjugate gradient for the linear system `A x = b`
  - `crank_nicolson()` — time integration loop; builds the implicit system each step
  - Visualization: `plot_source_and_steady_state()`, `plot_transient_timesteps()`, `generate_animation()`
- [project_1/project_1.ipynb](project_1/project_1.ipynb) — driver: sets up grid, BCs, and calls utils

### Project 2 — Navier-Stokes ([project_2/](project_2/))

- [project_2/operators.py](project_2/operators.py) — spatial operators on a **staggered MAC grid**:
  - `G()` — gradient (pressure cell centers → velocity faces)
  - `D()` — divergence (velocity faces → pressure cell centers)
  - `L()` — vector Laplacian with wall BC corrections
  - `A()` — advection operator (2nd-order finite differences; handles u and v separately)
  - `bc_D()` / `bc_L()` — boundary correction vectors for D and L operators
- [project_2/project_2.ipynb](project_2/project_2.ipynb) — fractional-step solver:
  1. **Advection + diffusion predictor** — compute intermediate velocity `u*` without enforcing continuity
  2. **Pressure Poisson** — solve `D G φ = D u* / Δt` via scipy CG
  3. **Velocity correction** — `u^{n+1} = u* - Δt G φ`, enforce divergence-free condition
- [project_2/operator_unit_tests.ipynb](project_2/operator_unit_tests.ipynb) — validates each operator independently (known-field MMS tests)

### MAC Grid Conventions (Project 2)

- **Pressure** defined at cell centers: `(i, j)` for `i ∈ [0, Nx-1]`, `j ∈ [0, Ny-1]`
- **u-velocity** defined at vertical cell faces (east/west): `(i+½, j)`
- **v-velocity** defined at horizontal cell faces (north/south): `(i, j+½)`
- All operators are assembled as `scipy.sparse` matrices operating on flattened 1D arrays in row-major (C) order
- Reference: [project_2/notes_and_refs/Notes_AllOperators.pdf](project_2/notes_and_refs/Notes_AllOperators.pdf) — canonical derivation of all operators

### Benchmark

Lid-driven cavity flow compared against Ghia et al. reference data in [project_2/notes_and_refs/Ref_Ghia.pdf](project_2/notes_and_refs/Ref_Ghia.pdf).
