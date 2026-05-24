#!/usr/bin/env python3
"""
24_scalar_curvature_grid4x4.py
================================================================
Scalar curvature R(β) on the Grid 4×4 along the h=0.01 path.

Purpose:
  - Do not expect N=16 curvature to locate an intrinsic window boundary.
  - Check whether finite-size R(β) is smooth along the h=0.01 path.
  - Verify coordinate invariance by computing R independently in
    (β,h) and canonical (β, B=βh) coordinates.
  - Plot R alongside Δ(β), which differs strongly across coordinates.

Method:
  1. Use a 9-point stencil to estimate g_ij and its first/second derivatives.
  2. Build Christoffel symbols and the Riemann tensor to compute scalar R.
  3. Repeat in both (β, h) and (β, B) coordinates as an implementation check.
  4. Compare with Δ(β) on the same path.

Implementation notes:
  - dβ = 0.002, dh = 0.0001 in (β,h); dB = 0.00005 in canonical coordinates.
  - Grid 4×4 only: N=16 gives 65536 states, small enough for exact sums.
================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pickle
import time

# Configuration
N = 16
J_val = 1.0
h_path = 0.01
beta_c = np.log(1.0 + np.sqrt(2.0)) / 2.0

# Scan path
beta_path = np.concatenate([
    np.linspace(0.10, 0.35, 25),
    np.linspace(0.36, 0.50, 35),  # denser near β_c
    np.linspace(0.51, 0.55, 8),
])
beta_path = np.sort(np.unique(np.round(beta_path, 5)))

# Finite-difference steps
dbeta = 0.002
dh    = 0.0001
dB    = 0.00005  # B perturbation in canonical coordinates

print("=" * 64)
print("Scalar curvature R(β) on Grid 4×4")
print("=" * 64)
print(f"β path: {len(beta_path)} points, [{beta_path[0]:.3f}, {beta_path[-1]:.3f}]")
print(f"Fixed h: {h_path}")
print(f"β_c (Onsager): {beta_c:.4f}")
print(f"Perturbations: dβ={dbeta}, dh={dh}, dB={dB}")

# ============================================================
# Grid 4×4 and observables
# ============================================================

def grid_2d_edges(n, periodic=True):
    """Grid edges.  periodic=True wraps around (torus), matching paper."""
    edges = []
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            # right neighbour
            c_next = (c + 1) % n if periodic else c + 1
            if periodic or c_next < n:
                edges.append((idx, r * n + c_next))
            # down neighbour
            r_next = (r + 1) % n if periodic else r + 1
            if periodic or r_next < n:
                edges.append((idx, r_next * n + c))
    return edges


edges = grid_2d_edges(4)
print(f"Grid 4×4: {len(edges)} edges")

n_states = 2 ** N
state_indices = np.arange(n_states, dtype=np.uint32)
spins = np.empty((n_states, N), dtype=np.int8)
for i in range(N):
    spins[:, i] = 2 * ((state_indices >> i) & 1) - 1
M_obs = np.sum(spins, axis=1, dtype=np.int16).astype(np.float64)
E_obs = np.zeros(n_states, dtype=np.int16)
for (i, j) in edges:
    E_obs += spins[:, i] * spins[:, j]
E_obs = E_obs.astype(np.float64)
print(f"Precomputed E_obs and M_obs with shape {E_obs.shape}")


# ============================================================
# FIM in the two coordinate systems
# ============================================================

def compute_fim_betah(beta, h):
    """Compute the FIM in (β, h) coordinates."""
    log_w = beta * (J_val * E_obs + h * M_obs)
    log_w -= log_w.max()
    w = np.exp(log_w)
    p = w / w.sum()
    H_eff = J_val * E_obs + h * M_obs
    mH = (H_eff * p).sum()
    mM = (M_obs * p).sum()
    var_H = ((H_eff - mH) ** 2 * p).sum()
    var_M = ((M_obs - mM) ** 2 * p).sum()
    cov_HM = ((H_eff - mH) * (M_obs - mM) * p).sum()
    return np.array([
        [var_H,         beta * cov_HM],
        [beta * cov_HM, beta ** 2 * var_M],
    ])


def compute_fim_canon(beta, B):
    """Compute the FIM in canonical (β, B=βh) coordinates.

    The manuscript fixes J=1, so the canonical sufficient statistics are E and M.
    """
    log_w = beta * E_obs + B * M_obs
    log_w -= log_w.max()
    w = np.exp(log_w)
    p = w / w.sum()
    mE = (E_obs * p).sum()
    mM = (M_obs * p).sum()
    var_E = ((E_obs - mE) ** 2 * p).sum()
    var_M = ((M_obs - mM) ** 2 * p).sum()
    cov_EM = ((E_obs - mE) * (M_obs - mM) * p).sum()
    return np.array([
        [var_E,  cov_EM],
        [cov_EM, var_M],
    ])


# ============================================================
# 9-point stencil for g, ∂g, and ∂²g
# ============================================================

def stencil_9pt(point_func, x0, y0, dx, dy):
    """
    point_func(x, y) -> 2x2 metric.
    Returns a dict with g, dg_dx, dg_dy, d2g_dxx, d2g_dyy, d2g_dxy.
    """
    g_pp = point_func(x0 + dx, y0 + dy)
    g_pm = point_func(x0 + dx, y0 - dy)
    g_mp = point_func(x0 - dx, y0 + dy)
    g_mm = point_func(x0 - dx, y0 - dy)
    g_p0 = point_func(x0 + dx, y0)
    g_m0 = point_func(x0 - dx, y0)
    g_0p = point_func(x0,      y0 + dy)
    g_0m = point_func(x0,      y0 - dy)
    g_00 = point_func(x0,      y0)

    dg_dx = (g_p0 - g_m0) / (2 * dx)
    dg_dy = (g_0p - g_0m) / (2 * dy)
    d2g_dxx = (g_p0 - 2 * g_00 + g_m0) / (dx ** 2)
    d2g_dyy = (g_0p - 2 * g_00 + g_0m) / (dy ** 2)
    d2g_dxy = (g_pp - g_pm - g_mp + g_mm) / (4 * dx * dy)
    return {
        "g": g_00, "dg_dx": dg_dx, "dg_dy": dg_dy,
        "d2g_dxx": d2g_dxx, "d2g_dyy": d2g_dyy, "d2g_dxy": d2g_dxy,
    }


# ============================================================
# Scalar curvature for a 2D Riemannian metric
# ============================================================

def scalar_curvature_2d(s):
    """Compute 2D scalar curvature R from stencil derivatives."""
    g = s["g"]
    dg = [s["dg_dx"], s["dg_dy"]]
    d2g = [[s["d2g_dxx"], s["d2g_dxy"]],
           [s["d2g_dxy"], s["d2g_dyy"]]]

    g_inv = np.linalg.inv(g)

    # Christoffel Γ^k_ij
    Gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                acc = 0.0
                for l in range(2):
                    acc += g_inv[k, l] * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                Gamma[k, i, j] = 0.5 * acc

    # ∂_m Γ^k_ij
    dGamma = np.zeros((2, 2, 2, 2))
    for m in range(2):
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    acc = 0.0
                    for l in range(2):
                        # ∂_m g^{kl} = -g^{ka} g^{lb} ∂_m g_{ab}
                        dginv_kl = 0.0
                        for a in range(2):
                            for b in range(2):
                                dginv_kl -= g_inv[k, a] * g_inv[l, b] * dg[m][a, b]
                        acc += dginv_kl * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                        acc += g_inv[k, l] * (
                            d2g[m][i][j, l] + d2g[m][j][i, l] - d2g[m][l][i, j]
                        )
                    dGamma[m, k, i, j] = 0.5 * acc

    # Riemann R^l_kij = ∂_i Γ^l_jk - ∂_j Γ^l_ik + Γ^l_im Γ^m_jk - Γ^l_jm Γ^m_ik
    Riemann = np.zeros((2, 2, 2, 2))
    for l in range(2):
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    val = dGamma[i, l, j, k] - dGamma[j, l, i, k]
                    for m in range(2):
                        val += Gamma[l, i, m] * Gamma[m, j, k]
                        val -= Gamma[l, j, m] * Gamma[m, i, k]
                    Riemann[l, k, i, j] = val

    # Ricci R_kj = sum_i R^i_kij
    Ricci = np.zeros((2, 2))
    for k in range(2):
        for j in range(2):
            for i in range(2):
                Ricci[k, j] += Riemann[i, k, i, j]

    # Scalar R = g^{kj} R_kj
    R_scalar = 0.0
    for k in range(2):
        for j in range(2):
            R_scalar += g_inv[k, j] * Ricci[k, j]
    return R_scalar


# ============================================================
# Main scan
# ============================================================

print("\n--- Computing R(β) along the path ---")
n_path = len(beta_path)
R_betah = np.full(n_path, np.nan)
R_canon = np.full(n_path, np.nan)

I_eff_betah = np.full(n_path, np.nan)
I_eff_canon = np.full(n_path, np.nan)

t0 = time.time()
for idx, beta in enumerate(beta_path):
    B_path = beta * h_path

    # (β, h) coordinates
    try:
        s_bh = stencil_9pt(compute_fim_betah, beta, h_path, dbeta, dh)
        R_betah[idx] = scalar_curvature_2d(s_bh)
        eigs_bh = np.linalg.eigvalsh(s_bh["g"])
        if eigs_bh[0] > 1e-14:
            I_eff_betah[idx] = np.log(eigs_bh[-1] / eigs_bh[0])
    except Exception as e:
        print(f"  ({idx + 1}/{n_path}) β={beta:.4f}: (β,h) mode failed -- {e}")

    # (β, B) coordinates
    try:
        s_can = stencil_9pt(compute_fim_canon, beta, B_path, dbeta, dB)
        R_canon[idx] = scalar_curvature_2d(s_can)
        eigs_can = np.linalg.eigvalsh(s_can["g"])
        if eigs_can[0] > 1e-14:
            I_eff_canon[idx] = np.log(eigs_can[-1] / eigs_can[0])
    except Exception as e:
        print(f"  ({idx + 1}/{n_path}) β={beta:.4f}: canonical mode failed -- {e}")

    if (idx + 1) % 10 == 0 or idx == 0 or idx == n_path - 1:
        elapsed = time.time() - t0
        print(f"  [{idx + 1}/{n_path}] β={beta:.4f}  R_(β,h)={R_betah[idx]:+.3e}  "
              f"R_canon={R_canon[idx]:+.3e}  ({elapsed:.1f}s elapsed)")

print(f"Total time: {time.time() - t0:.1f}s")

# ============================================================
# Check coordinate invariance of R
# ============================================================

valid = np.isfinite(R_betah) & np.isfinite(R_canon)
if valid.sum() > 5:
    diff = R_betah[valid] - R_canon[valid]
    rel_err = np.abs(diff) / (np.abs(R_betah[valid]) + 1e-10)
    print("\n--- Coordinate-invariance check ---")
    print(f"  Valid points: {valid.sum()}/{n_path}")
    print(f"  Median |R_(β,h) - R_canon|: {np.median(np.abs(diff)):.3e}")
    print(f"  Median relative error: {np.median(rel_err):.3e}")
    print(f"  95th-percentile relative error: {np.quantile(rel_err, 0.95):.3e}")
    if np.median(rel_err) < 1e-2:
        print("  PASS: R agrees across coordinates within numerical precision")
    else:
        print("  WARNING: R differs across coordinates; check stencil steps or implementation")

# ============================================================
# Numerical derivative for Δ(β)
# ============================================================

def cdiff(y, x):
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy


Delta_bh = -cdiff(I_eff_betah, beta_path)
Delta_canon = -cdiff(I_eff_canon, beta_path)

# ============================================================
# Summary facts
# ============================================================

print("\n" + "=" * 64)
print("Summary facts")
print("=" * 64)

# R near β_c
near_bc = np.argmin(np.abs(beta_path - beta_c))
print(f"\nFact 1: R(β) near β_c={beta_c:.4f}")
window = slice(max(0, near_bc - 3), min(n_path, near_bc + 4))
for k in range(window.start, window.stop):
    print(f"  β={beta_path[k]:.4f}: R_(β,h)={R_betah[k]:+.3e}  R_canon={R_canon[k]:+.3e}  "
          f"Δ_(β,h)={Delta_bh[k]:+.3f}  Δ_canon={Delta_canon[k]:+.3f}")

print(f"\nFact 2: R extrema vs β_c={beta_c:.4f}")
for label, R_arr in [("R_(β,h)", R_betah), ("R_canon", R_canon)]:
    valid_R = np.isfinite(R_arr)
    if valid_R.any():
        idx_min = np.argmin(np.where(valid_R, R_arr, np.inf))
        idx_absmax = np.argmax(np.where(valid_R, np.abs(R_arr), -np.inf))
        print(f"  {label}: argmin@β={beta_path[idx_min]:.4f} (R={R_arr[idx_min]:+.3e}); "
              f"|R|_max@β={beta_path[idx_absmax]:.4f} (R={R_arr[idx_absmax]:+.3e})")

# Compare with the Δ zero crossing in (β,h)
def find_zero(y, x):
    sign_y = np.sign(y)
    for k in range(len(y) - 1):
        if sign_y[k] > 0 > sign_y[k + 1]:
            if y[k + 1] != y[k]:
                return x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
            return 0.5 * (x[k] + x[k + 1])
    return np.nan


zero_bh = find_zero(Delta_bh, beta_path)
zero_canon = find_zero(Delta_canon, beta_path)
print("\nFact 3: First + to - zero crossing of Δ(β)")
print(f"  (β,h) mode: β*_G = {zero_bh}")
print(f"  canonical: β*_G = {zero_canon}")
print("  Compare the R extrema reported above with β*_G.")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: R(β), comparing both coordinates
ax = axes[0, 0]
ax.plot(beta_path, R_betah, "C0-", lw=1.6, label="$R$ from $(\\beta, h)$ stencil")
ax.plot(beta_path, R_canon, "C3--", lw=1.6, label="$R$ from $(\\beta, B)$ canonical stencil")
ax.axvline(beta_c, color="black", ls=":", alpha=0.7, label=f"$\\beta_c$={beta_c:.4f}")
ax.set_xlabel("β"); ax.set_ylabel("Scalar curvature $R$")
ax.set_title("Panel 1: Scalar curvature $R(β)$ along $h=0.01$ (Grid 4×4)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: |R| on log scale
ax = axes[0, 1]
ax.plot(beta_path, np.abs(R_betah), "C0-", lw=1.6, label="$|R|$ from $(\\beta, h)$")
ax.plot(beta_path, np.abs(R_canon), "C3--", lw=1.6, label="$|R|$ from canonical")
ax.axvline(beta_c, color="black", ls=":", alpha=0.7)
ax.set_yscale("log")
ax.set_xlabel("β"); ax.set_ylabel("$|R|$")
ax.set_title("Panel 2: $|R(β)|$ in log scale (look for singularity at $β_c$)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# Panel 3: I_eff and Δ in both coordinates
ax = axes[1, 0]
ax.plot(beta_path, I_eff_betah, "C0-", lw=1.5, label="$I_{eff}$ in $(β,h)$")
ax.plot(beta_path, I_eff_canon, "C3--", lw=1.5, label="$I_{eff}$ in canonical")
ax.axvline(beta_c, color="black", ls=":", alpha=0.5)
ax.set_xlabel("β"); ax.set_ylabel("$I_{eff} = \\log κ$")
ax.set_title("Panel 3: $I_{eff}(β)$ — clearly parameterization-dependent")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 4: R vs Δ — coordinate-invariant vs coordinate-dependent
ax = axes[1, 1]
ax2 = ax.twinx()
l1 = ax.plot(beta_path, np.abs(R_betah), "C2-", lw=2.0,
             label=r"$|R|$ (coord-invariant)")
l2 = ax2.plot(beta_path, Delta_bh, "C0--", lw=1.4, alpha=0.85, label="$Δ$ in $(β,h)$")
l3 = ax2.plot(beta_path, Delta_canon, "C3:", lw=1.4, alpha=0.85, label="$Δ$ in canonical")
ax2.axhline(0, color="black", lw=0.6, alpha=0.5)
ax.axvline(beta_c, color="black", ls=":", alpha=0.5)
ax.set_xlabel("β")
ax.set_ylabel("$|R|$", color="C2")
ax2.set_ylabel("$Δ(β)$", color="black")
ax.set_yscale("log")
ax.set_title("Panel 4: invariant $R$ vs parameterization-dependent $Δ$")
lines = l1 + l2 + l3
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)
OUT_PNG = os.path.join(fig_dir, "24_scalar_curvature_grid4x4.png")
plt.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
plt.close()
print(f"\nSaved figure: {OUT_PNG}")

# ============================================================
# Save outputs
# ============================================================

cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', '.cache_n16_curvature')
os.makedirs(cache_dir, exist_ok=True)
out_pkl = os.path.join(cache_dir, "results.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump({
        "N": N, "topology": "Grid 4x4", "h_path": h_path, "beta_c": beta_c,
        "dbeta": dbeta, "dh": dh, "dB": dB,
        "beta_path": beta_path,
        "R_betah": R_betah, "R_canon": R_canon,
        "I_eff_betah": I_eff_betah, "I_eff_canon": I_eff_canon,
        "Delta_betah": Delta_bh, "Delta_canon": Delta_canon,
    }, f)
print(f"Cache saved: {out_pkl}")

print("\n" + "=" * 64)
print("Scalar-curvature run complete.")
print("=" * 64)
