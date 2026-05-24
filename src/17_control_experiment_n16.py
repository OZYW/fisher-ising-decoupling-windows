#!/usr/bin/env python3
"""
Control experiment: N=16 calculations for 3-topology and 6-topology sets.

Purpose: isolate topology-set effects from system-size effects.
- N=16(3t): Grid 4x4, Ring, Complete
- N=16(6t): Grid 4x4, Ring, Complete, Star, Random Regular degree 4, Small World(k=4,p=0.3)

This legacy control compares against the corresponding N=25 experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import time

N = 16
J_val = 1.0
h_val = 0.01
beta_c = np.log(1.0 + np.sqrt(2.0)) / 2.0

# Same beta sampling as the N=25 experiment
beta_vals = np.concatenate([
    np.linspace(0.25, 0.35, 8),
    np.linspace(0.36, 0.52, 30),
    np.linspace(0.53, 0.65, 12),
])
beta_vals = np.sort(np.unique(np.round(beta_vals, 4)))

print("=" * 60)
print(f"Control experiment: N={N}")
print("=" * 60)
print(f"State-space size: 2^{N} = {2**N:,}")
print(f"Onsager critical inverse temperature: β_c = {beta_c:.4f}")
print(f"Number of beta samples: {len(beta_vals)}")

# ============================================================
# Topology definitions, reusing the N=25 parameters where applicable
# ============================================================

def grid_2d_edges(n, periodic=True):
    """Grid edges.  periodic=True wraps around (torus), matching paper. """
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

def ring_edges(N):
    return [(i, (i + 1) % N) for i in range(N)]

def complete_edges(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def star_edges(N):
    return [(0, i) for i in range(1, N)]

def random_regular_edges(N, d, seed=42):
    rng = np.random.default_rng(seed)
    if (N * d) % 2 != 0:
        raise ValueError("N*d must be even for a d-regular graph")
    base_stubs = np.repeat(np.arange(N), d)
    for _ in range(20000):
        stubs = base_stubs.copy()
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for a, b in stubs.reshape(-1, 2):
            if a == b:
                ok = False
                break
            edge = tuple(sorted((int(a), int(b))))
            if edge in edges:
                ok = False
                break
            edges.add(edge)
        if ok:
            degrees = np.zeros(N, dtype=int)
            for i, j in edges:
                degrees[i] += 1
                degrees[j] += 1
            if np.all(degrees == d):
                return sorted(edges)
    raise RuntimeError(f"failed to generate a simple {d}-regular graph on {N} nodes")

def small_world_edges(N, k=4, p=0.3, seed=42):
    rng = np.random.default_rng(seed)
    edges = set()
    half_k = k // 2
    for i in range(N):
        for j in range(1, half_k + 1):
            edges.add(tuple(sorted((i, (i + j) % N))))
    new_edges = set(edges)
    for i, j in sorted(edges):
        if rng.random() < p:
            new_edges.discard(tuple(sorted((i, j))))
            candidates = [x for x in range(N) if x != i and tuple(sorted((i, x))) not in new_edges]
            if candidates:
                new_j = rng.choice(candidates)
                new_edges.add(tuple(sorted((i, new_j))))
    return sorted(new_edges)

def compute_lambda2(N, edges):
    """Unweighted graph Laplacian spectral gap."""
    A = np.zeros((N, N))
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = np.linalg.eigvalsh(L)
    return np.sort(eigvals)[1]

# ============================================================
# Precompute observables. N=16 is small enough to avoid chunking.
# ============================================================

def precompute_ising_observables(N, edges):
    n_states = 2 ** N
    state_indices = np.arange(n_states, dtype=np.uint32)
    spins = np.empty((n_states, N), dtype=np.int8)
    for i in range(N):
        spins[:, i] = 2 * ((state_indices >> i) & 1) - 1
    M_obs = np.sum(spins, axis=1, dtype=np.int16)
    E_obs = np.zeros(n_states, dtype=np.int16)
    for (i, j) in edges:
        E_obs += spins[:, i] * spins[:, j]
    return E_obs, M_obs

def compute_results_for_topologies(N, topologies, beta_vals):
    """Compute results over all beta values for a topology set."""
    # Precompute observables
    E_obs_dict = {}
    M_obs_dict = {}
    for name, edges in topologies.items():
        E_obs, M_obs = precompute_ising_observables(N, edges)
        E_obs_dict[name] = E_obs
        M_obs_dict[name] = M_obs
        print(f"  Precomputed: {name} ({len(edges)} edges)")

    # Lambda2
    lambda2s = {name: compute_lambda2(N, edges) for name, edges in topologies.items()}
    for name, l2 in lambda2s.items():
        print(f"  {name}: λ₂ = {l2:.4f}")

    # Scan beta
    print(f"\nScanning {len(beta_vals)} beta values...")
    results = []

    for idx, beta in enumerate(beta_vals):
        loop_start = time.time()
        conds = []
        traces = []
        names = []

        for name, edges in topologies.items():
            E_obs = E_obs_dict[name]
            M_obs = M_obs_dict[name]

            log_weights = beta * (J_val * E_obs + h_val * M_obs)
            max_logw = np.max(log_weights)
            weights = np.exp(log_weights - max_logw)
            probs = weights / weights.sum()

            H_eff = J_val * E_obs + h_val * M_obs
            mean_H = np.dot(H_eff, probs)
            mean_M = np.dot(M_obs, probs)

            var_H = np.dot((H_eff - mean_H) ** 2, probs)
            var_M = np.dot((M_obs - mean_M) ** 2, probs)
            cov_HM = np.dot((H_eff - mean_H) * (M_obs - mean_M), probs)

            I_bb = var_H
            I_bh = beta * cov_HM
            I_hh = beta ** 2 * var_M

            fim = np.array([[I_bb, I_bh], [I_bh, I_hh]])
            eigs = np.linalg.eigvalsh(fim)

            if eigs[0] > 1e-12:
                cond = eigs[-1] / eigs[0]
            else:
                cond = np.nan

            conds.append(cond)
            traces.append(I_bb + I_hh)
            names.append(name)

        conds_arr = np.array(conds)
        lambda2s_arr = np.array([lambda2s[n] for n in names])
        valid = ~np.isnan(conds_arr) & (conds_arr > 0)
        if valid.sum() >= 2:
            corr_cond = np.corrcoef(np.log10(conds_arr[valid]), lambda2s_arr[valid])[0, 1]
            corr_trace = np.corrcoef(np.array(traces)[valid], lambda2s_arr[valid])[0, 1]
        else:
            corr_cond = corr_trace = np.nan

        results.append({
            'beta': beta,
            'corr_cond': corr_cond,
            'corr_trace': corr_trace,
            'conds': conds_arr.copy(),
            'traces': np.array(traces).copy(),
        })

        elapsed = time.time() - loop_start
        print(f"  [{idx+1}/{len(beta_vals)}] β={beta:.3f}: corr_cond={corr_cond:+.4f}, elapsed={elapsed:.2f}s")

    return results, lambda2s

# ============================================================
# Experiment 1: N=16, 3 topologies
# ============================================================

print("\n" + "=" * 60)
print("Experiment 1: N=16, 3 topologies (Grid 4x4, Ring, Complete)")
print("=" * 60)

topologies_3t = {
    'Grid 4x4': grid_2d_edges(4),
    'Ring': ring_edges(N),
    'Complete': complete_edges(N),
}

results_3t, lambda2s_3t = compute_results_for_topologies(N, topologies_3t, beta_vals)

# ============================================================
# Experiment 2: N=16, 6 topologies
# ============================================================

print("\n" + "=" * 60)
print("Experiment 2: N=16, 6 topologies (Grid 4x4, Ring, Complete, Star, RR, SW)")
print("=" * 60)

topologies_6t = {
    'Grid 4x4': grid_2d_edges(4),
    'Ring': ring_edges(N),
    'Complete': complete_edges(N),
    'Star': star_edges(N),
    'Random Regular (degree 4)': random_regular_edges(N, 4),
    'Small World': small_world_edges(N, k=4, p=0.3),
}

results_6t, lambda2s_6t = compute_results_for_topologies(N, topologies_6t, beta_vals)

# ============================================================
# Peak analysis
# ============================================================

def analyze_peak(results, label):
    betas = np.array([r['beta'] for r in results])
    corrs = np.array([r['corr_cond'] for r in results])
    valid = ~np.isnan(corrs)
    peak_idx = np.argmax(corrs[valid])
    peak_beta = betas[valid][peak_idx]
    peak_corr = corrs[valid][peak_idx]

    high_mask = valid & (corrs > 0.95)
    if np.any(high_mask):
        ws, we = betas[high_mask].min(), betas[high_mask].max()
    else:
        ws = we = np.nan

    print(f"\n{label}:")
    print(f"  Peak β = {peak_beta:.4f}, corr = {peak_corr:.4f}")
    print(f"  Offset |β_peak - β_c| = {abs(peak_beta - beta_c):.4f}")
    if not np.isnan(ws):
        print(f"  High-correlation window (corr>0.95): β ∈ [{ws:.3f}, {we:.3f}]")

    return peak_beta, peak_corr, ws, we

print("\n" + "=" * 60)
print("Peak analysis")
print("=" * 60)

peak_3t, corr_3t, ws3, we3 = analyze_peak(results_3t, "N=16 (3 topologies)")
peak_6t, corr_6t, ws6, we6 = analyze_peak(results_6t, "N=16 (6 topologies)")

# ============================================================
# Cross-experiment comparison table
# ============================================================

print("\n" + "=" * 60)
print("Cross-experiment comparison")
print("=" * 60)
print(f"{'Experiment':<24s} {'Peak β':<10s} {'Offset':<12s} {'Peak corr':<12s} {'Window width':<12s}")
print("-" * 70)
print(f"{'N=16 old 7-topology data':<24s} {'0.5493':<10s} {'0.1086':<12s} {'0.9983':<12s} {'0.472':<12s}")
print(f"{'N=16 3-topology control':<24s} {f'{peak_3t:.4f}':<10s} {f'{abs(peak_3t-beta_c):.4f}':<12s} {f'{corr_3t:.4f}':<12s} {f'{we3-ws3:.3f}':<12s}")
print(f"{'N=16 6-topology control':<24s} {f'{peak_6t:.4f}':<10s} {f'{abs(peak_6t-beta_c):.4f}':<12s} {f'{corr_6t:.4f}':<12s} {f'{we6-ws6:.3f}':<12s}")
print(f"{'N=25 3 topologies':<24s} {'0.3500':<10s} {'0.0907':<12s} {'1.0000':<12s} {'0.400':<12s}")
print(f"{'N=25 6 topologies':<24s} {'0.4317':<10s} {'0.0090':<12s} {'0.9978':<12s} {'0.386':<12s}")

# ============================================================
# Controlled comparisons
# ============================================================

print("\n" + "=" * 60)
print("Controlled-variable analysis")
print("=" * 60)

closer_a = "toward β_c" if abs(0.3500 - beta_c) < abs(peak_3t - beta_c) else "away from β_c"
closer_b = "toward β_c" if abs(0.4317 - beta_c) < abs(peak_6t - beta_c) else "away from β_c"

# Same topology set, different system sizes
print("\nComparison A: same topology set (3t: Grid+Ring+Complete), different sizes:")
print(f"  N=16(3t) -> N=25(3t): peak moves from {peak_3t:.4f} to 0.3500")
print(f"  Shift: {0.3500 - peak_3t:+.4f} ({closer_a})")

print("\nComparison B: same topology set (6t), different sizes:")
print(f"  N=16(6t) -> N=25(6t): peak moves from {peak_6t:.4f} to 0.4317")
print(f"  Shift: {0.4317 - peak_6t:+.4f} ({closer_b})")

print("\nComparison C: same size (N=16), different topology sets:")
print(f"  N=16(3t) -> N=16(6t): peak moves from {peak_3t:.4f} to {peak_6t:.4f}")
print(f"  Topology-set effect: {peak_6t - peak_3t:+.4f}")

print("\nComparison D: old data vs control experiment:")
print(f"  N=16 old 7-topology data vs N=16 new 6-topology control: 0.5493 vs {peak_6t:.4f}")
print(f"  Difference: {peak_6t - 0.5493:+.4f}")

# ============================================================
# Save results
# ============================================================

cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', '.cache_n16_control')
os.makedirs(cache_dir, exist_ok=True)

with open(os.path.join(cache_dir, 'results_n16_control.pkl'), 'wb') as f:
    pickle.dump({
        'N': N,
        'beta_c': beta_c,
        'results_3t': results_3t,
        'results_6t': results_6t,
        'lambda2s_3t': lambda2s_3t,
        'lambda2s_6t': lambda2s_6t,
        'peak_3t': peak_3t,
        'peak_6t': peak_6t,
    }, f)

print(f"\nResults saved to: {cache_dir}/results_n16_control.pkl")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

betas = np.array([r['beta'] for r in results_3t])

# Subplot 1: 3 topologies vs 6 topologies (N=16)
ax = axes[0]
corrs_3t_arr = np.array([r['corr_cond'] for r in results_3t])
corrs_6t_arr = np.array([r['corr_cond'] for r in results_6t])
ax.plot(betas, corrs_3t_arr, 'b-', linewidth=2, label=f'N=16 (3t), peak={peak_3t:.3f}')
ax.plot(betas, corrs_6t_arr, 'g-', linewidth=2, label=f'N=16 (6t), peak={peak_6t:.3f}')
ax.axvline(x=beta_c, color='r', linestyle='--', alpha=0.7, label=f'β_c={beta_c:.3f}')
ax.set_xlabel('β', fontsize=12)
ax.set_ylabel('Correlation', fontsize=12)
ax.set_title('N=16: Topology Effect', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.85, 1.0)

# Subplot 2: 6 topologies across sizes
ax = axes[1]
ax.plot(betas, corrs_6t_arr, 'g-', linewidth=2.5, label=f'N=16 (6t), peak={peak_6t:.3f}')
# N=25(6t) data (optional; skip gracefully if not present)
n25_path = os.path.join(os.path.dirname(__file__), '..', 'data', '.cache_n25', 'checkpoint_n25_6topo.pkl')
if os.path.exists(n25_path):
    with open(n25_path, 'rb') as f:
        chk25 = pickle.load(f)
    betas25 = np.array([r['beta'] for r in chk25['results']])
    corrs25 = np.array([r['corr_cond'] for r in chk25['results']])
    ax.plot(betas25, corrs25, 'r-', linewidth=2.5, label=f'N=25 (6t), peak=0.432')
else:
    ax.text(0.5, 0.5, 'N=25 data not available\n(run N=25 scan separately)',
            transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
ax.axvline(x=beta_c, color='r', linestyle='--', alpha=0.7, label=f'β_c={beta_c:.3f}')
ax.set_xlabel('β', fontsize=12)
ax.set_ylabel('Correlation', fontsize=12)
ax.set_title('Cross-Size (same 6 topologies)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.85, 1.0)

plt.tight_layout()
fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, '17_control_experiment_results.png'), dpi=150, bbox_inches='tight')
print("\nSaved figure: figures/17_control_experiment_results.png")

print("\n" + "=" * 60)
print("Control experiment complete")
print("=" * 60)
