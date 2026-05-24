#!/usr/bin/env python3
"""
Exact-enumeration scan for the decoupling-window manuscript.

This script is the deterministic source for the main condition-number evidence:

  * main path: h = 0.01, interpreted in the paper as a weak-field finite-size
    diagnostic rather than the zero-field critical Ising model;
  * robustness path: h = 0, computed in both P1=(beta,h) and P2=(beta,B).

The output figure and CSV/NPZ files are written under ../figures and ../data.
"""

from __future__ import annotations

import csv
import os
import pickle
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N = 16
J_VAL = 1.0
H_MAIN = 0.01
H_ROBUST = 0.0
BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0
BETA_GRID = np.linspace(0.05, 0.55, 401)


@dataclass(frozen=True)
class ScanResult:
    h: float
    beta: np.ndarray
    cond_p1: np.ndarray
    cond_p2: np.ndarray
    ieff_p1: np.ndarray
    ieff_p2: np.ndarray
    delta_p1: np.ndarray
    delta_p2: np.ndarray
    beta_star_p1: np.ndarray
    beta_star_p2: np.ndarray


def grid_2d_edges(n: int, periodic: bool = True) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            c_next = (c + 1) % n if periodic else c + 1
            if periodic or c_next < n:
                edges.append((idx, r * n + c_next))
            r_next = (r + 1) % n if periodic else r + 1
            if periodic or r_next < n:
                edges.append((idx, r_next * n + c))
    return edges


def ring_edges(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def complete_edges(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def star_edges(n: int) -> list[tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def random_regular_edges(n: int, d: int, seed: int = 42) -> list[tuple[int, int]]:
    """Generate a simple undirected d-regular graph by pairing shuffled stubs."""
    if (n * d) % 2 != 0:
        raise ValueError("n*d must be even for a d-regular graph")
    rng = np.random.default_rng(seed)
    base_stubs = np.repeat(np.arange(n), d)
    for _ in range(20_000):
        stubs = base_stubs.copy()
        rng.shuffle(stubs)
        edges: set[tuple[int, int]] = set()
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
            degrees = np.zeros(n, dtype=int)
            for i, j in edges:
                degrees[i] += 1
                degrees[j] += 1
            if np.all(degrees == d):
                return sorted(edges)
    raise RuntimeError(f"failed to generate a simple {d}-regular graph on {n} nodes")


def small_world_edges(n: int, k: int = 4, p: float = 0.3, seed: int = 42) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    half_k = k // 2
    for i in range(n):
        for j in range(1, half_k + 1):
            edges.add(tuple(sorted((i, (i + j) % n))))
    new_edges = set(edges)
    for i, j in sorted(edges):
        if rng.random() < p:
            new_edges.discard((i, j))
            candidates = [
                x
                for x in range(n)
                if x != i and tuple(sorted((i, x))) not in new_edges
            ]
            if candidates:
                new_j = int(rng.choice(candidates))
                new_edges.add(tuple(sorted((i, new_j))))
    return sorted(new_edges)


def compute_lambda2(n: int, edges: list[tuple[int, int]]) -> float:
    """Unweighted graph Laplacian spectral gap."""
    adj = np.zeros((n, n), dtype=float)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    lap = np.diag(adj.sum(axis=1)) - adj
    eigvals = np.linalg.eigvalsh(lap)
    return float(np.sort(eigvals)[1])


def precompute_observables(n: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    n_states = 2**n
    state_indices = np.arange(n_states, dtype=np.uint32)
    spins = np.empty((n_states, n), dtype=np.int8)
    for i in range(n):
        spins[:, i] = 2 * ((state_indices >> i) & 1) - 1
    magnetization = np.sum(spins, axis=1, dtype=np.int16).astype(np.float64)
    edge_sum = np.zeros(n_states, dtype=np.int16)
    for i, j in edges:
        edge_sum += spins[:, i] * spins[:, j]
    return edge_sum.astype(np.float64), magnetization


def weighted_moments(beta: float, h: float, edge_sum: np.ndarray, magnetization: np.ndarray):
    interaction = J_VAL * edge_sum
    beta_score_stat = interaction + h * magnetization
    log_weights = beta * beta_score_stat
    log_weights -= np.max(log_weights)
    probs = np.exp(log_weights)
    probs /= probs.sum()

    mean_s = np.dot(beta_score_stat, probs)
    mean_e = np.dot(interaction, probs)
    mean_m = np.dot(magnetization, probs)

    centered_s = beta_score_stat - mean_s
    centered_e = interaction - mean_e
    centered_m = magnetization - mean_m

    var_s = np.dot(centered_s * centered_s, probs)
    var_e = np.dot(centered_e * centered_e, probs)
    var_m = np.dot(centered_m * centered_m, probs)
    cov_sm = np.dot(centered_s * centered_m, probs)
    cov_em = np.dot(centered_e * centered_m, probs)
    return var_s, var_e, var_m, cov_sm, cov_em


def fim_pair(beta: float, h: float, edge_sum: np.ndarray, magnetization: np.ndarray):
    """Return FIMs in P1=(beta,h) and P2=(beta,B=beta*h)."""
    var_s, var_e, var_m, cov_sm, cov_em = weighted_moments(
        beta, h, edge_sum, magnetization
    )
    fim_p1 = np.array(
        [
            [var_s, beta * cov_sm],
            [beta * cov_sm, beta * beta * var_m],
        ],
        dtype=float,
    )
    fim_p2 = np.array(
        [
            [var_e, cov_em],
            [cov_em, var_m],
        ],
        dtype=float,
    )
    return fim_p1, fim_p2


def cond_number(fim: np.ndarray, eps: float = 1e-12) -> float:
    eigvals = np.linalg.eigvalsh(fim)
    if eigvals[0] <= eps:
        return np.nan
    return float(eigvals[-1] / eigvals[0])


def central_diff(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    deriv = np.zeros_like(values)
    deriv[1:-1] = (values[2:] - values[:-2]) / (x[2:, None] - x[:-2, None])
    deriv[0] = (values[1] - values[0]) / (x[1] - x[0])
    deriv[-1] = (values[-1] - values[-2]) / (x[-1] - x[-2])
    return deriv


def first_pos_to_neg(delta: np.ndarray, beta: np.ndarray) -> float:
    for k in range(len(delta) - 1):
        if delta[k] > 0 and delta[k + 1] < 0:
            d0, d1 = delta[k], delta[k + 1]
            b0, b1 = beta[k], beta[k + 1]
            if d1 == d0:
                return float(0.5 * (b0 + b1))
            return float(b0 - d0 * (b1 - b0) / (d1 - d0))
    return np.nan


def run_scan(h: float, topo_names: list[str], obs_cache: dict[str, tuple[np.ndarray, np.ndarray]]):
    n_beta = len(BETA_GRID)
    n_topo = len(topo_names)
    cond_p1 = np.full((n_beta, n_topo), np.nan)
    cond_p2 = np.full((n_beta, n_topo), np.nan)
    t0 = time.time()
    print(f"\n--- Exact scan h={h:g}: {n_beta} beta points x {n_topo} topologies ---")
    for i, beta in enumerate(BETA_GRID):
        for j, name in enumerate(topo_names):
            edge_sum, magnetization = obs_cache[name]
            fim_p1, fim_p2 = fim_pair(beta, h, edge_sum, magnetization)
            cond_p1[i, j] = cond_number(fim_p1)
            cond_p2[i, j] = cond_number(fim_p2)
        if i == 0 or (i + 1) % 100 == 0 or i == n_beta - 1:
            print(f"  [{i + 1:3d}/{n_beta}] beta={beta:.4f} elapsed={time.time() - t0:.1f}s")

    ieff_p1 = np.log(cond_p1)
    ieff_p2 = np.log(cond_p2)
    delta_p1 = -central_diff(ieff_p1, BETA_GRID)
    delta_p2 = -central_diff(ieff_p2, BETA_GRID)
    beta_star_p1 = np.array([first_pos_to_neg(delta_p1[:, j], BETA_GRID) for j in range(n_topo)])
    beta_star_p2 = np.array([first_pos_to_neg(delta_p2[:, j], BETA_GRID) for j in range(n_topo)])
    return ScanResult(
        h=h,
        beta=BETA_GRID.copy(),
        cond_p1=cond_p1,
        cond_p2=cond_p2,
        ieff_p1=ieff_p1,
        ieff_p2=ieff_p2,
        delta_p1=delta_p1,
        delta_p2=delta_p2,
        beta_star_p1=beta_star_p1,
        beta_star_p2=beta_star_p2,
    )


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return np.nan, np.nan, int(mask.sum())
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    slope = float(np.polyfit(x[mask], y[mask], 1)[0])
    return r, slope, int(mask.sum())


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def summarize(
    label: str,
    scan: ScanResult,
    lambda2_arr: np.ndarray,
    topo_names: list[str],
    non_complete: list[int],
) -> None:
    print(f"\n=== {label} (h={scan.h:g}) ===")
    print(f"{'topology':<16} {'lambda2':>9} {'P1 beta*':>10} {'P2 beta*':>10}")
    for j, name in enumerate(topo_names):
        print(
            f"{name:<16} {lambda2_arr[j]:9.4f} "
            f"{scan.beta_star_p1[j]:10.4f} {scan.beta_star_p2[j]:10.4f}"
        )
    for pname, beta_star in [("P1=(beta,h)", scan.beta_star_p1), ("P2=(beta,B)", scan.beta_star_p2)]:
        vals = beta_star[non_complete]
        valid = vals[np.isfinite(vals)]
        r, slope, n = safe_pearson(lambda2_arr[non_complete], vals)
        ci = fisher_ci(r, n)
        if len(valid):
            print(
                f"  {pname:<13} mean={valid.mean():.4f}, std={valid.std(ddof=0):.4f}, "
                f"crossings={len(valid)}/5, r={r:+.4f}, slope={slope:+.4f}, "
                f"95% CI=({ci[0]:+.3f}, {ci[1]:+.3f})"
            )


def write_summary_csv(
    path: str,
    scans: list[ScanResult],
    topo_names: list[str],
    lambda2s: dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["h", "parameterization", "topology", "lambda2", "beta_star"])
        for scan in scans:
            for param, beta_star in [("P1_beta_h", scan.beta_star_p1), ("P2_beta_B", scan.beta_star_p2)]:
                for name, value in zip(topo_names, beta_star):
                    writer.writerow([scan.h, param, name, lambda2s[name], value])


def save_npz(path: str, scans: list[ScanResult], topo_names: list[str], lambda2_arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "beta": BETA_GRID,
        "topo_names": np.array(topo_names),
        "lambda2": lambda2_arr,
        "beta_c": BETA_C,
    }
    for scan in scans:
        suffix = "h001" if scan.h == H_MAIN else "h0"
        payload[f"{suffix}_cond_p1"] = scan.cond_p1
        payload[f"{suffix}_cond_p2"] = scan.cond_p2
        payload[f"{suffix}_ieff_p1"] = scan.ieff_p1
        payload[f"{suffix}_ieff_p2"] = scan.ieff_p2
        payload[f"{suffix}_delta_p1"] = scan.delta_p1
        payload[f"{suffix}_delta_p2"] = scan.delta_p2
        payload[f"{suffix}_beta_star_p1"] = scan.beta_star_p1
        payload[f"{suffix}_beta_star_p2"] = scan.beta_star_p2
    np.savez_compressed(path, **payload)


def make_figure(
    path: str,
    main: ScanResult,
    robust: ScanResult,
    topo_names: list[str],
    lambda2_arr: np.ndarray,
    non_complete: list[int],
) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(topo_names)))
    fig = plt.figure(figsize=(17, 12))

    ax = plt.subplot(3, 2, 1)
    for j, name in enumerate(topo_names):
        ax.plot(main.beta, main.ieff_p1[:, j], lw=1.4, color=colors[j], label=name)
    ax.axvline(BETA_C, color="black", ls="--", lw=1, alpha=0.55)
    ax.set_title("P1=(beta,h), h=0.01: I_eff")
    ax.set_xlabel("beta")
    ax.set_ylabel("log kappa(F)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = plt.subplot(3, 2, 2)
    for j, name in enumerate(topo_names):
        ax.plot(main.beta, main.ieff_p2[:, j], lw=1.4, color=colors[j], label=name)
    ax.axvline(BETA_C, color="black", ls="--", lw=1, alpha=0.55)
    ax.set_title("P2=(beta,B), h=0.01 path: I_eff")
    ax.set_xlabel("beta")
    ax.set_ylabel("log kappa(F)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = plt.subplot(3, 2, 3)
    for j, name in enumerate(topo_names):
        ax.plot(main.beta, main.delta_p1[:, j], lw=1.4, color=colors[j], label=name)
        if np.isfinite(main.beta_star_p1[j]):
            ax.axvline(main.beta_star_p1[j], color=colors[j], ls=":", lw=1, alpha=0.55)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(BETA_C, color="black", ls="--", lw=1, alpha=0.55)
    ax.set_title("P1=(beta,h), h=0.01: Delta and beta*")
    ax.set_xlabel("beta")
    ax.set_ylabel("Delta=-d log kappa / d beta")
    ax.grid(alpha=0.3)

    ax = plt.subplot(3, 2, 4)
    for j, name in enumerate(topo_names):
        ax.plot(main.beta, main.delta_p2[:, j], lw=1.4, color=colors[j], label=name)
        if np.isfinite(main.beta_star_p2[j]):
            ax.axvline(main.beta_star_p2[j], color=colors[j], ls=":", lw=1, alpha=0.55)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(BETA_C, color="black", ls="--", lw=1, alpha=0.55)
    ax.set_title("P2=(beta,B), h=0.01 path: Delta and beta*")
    ax.set_xlabel("beta")
    ax.set_ylabel("Delta=-d log kappa / d beta")
    ax.grid(alpha=0.3)

    ax = plt.subplot(3, 2, 5)
    for label, y, marker, color in [
        ("P1 h=0.01", main.beta_star_p1, "o", "C0"),
        ("P2 h=0.01", main.beta_star_p2, "s", "C1"),
    ]:
        valid = np.isfinite(y)
        ax.scatter(lambda2_arr[valid], y[valid], s=75, marker=marker, color=color,
                   edgecolor="black", linewidth=0.5, label=label)
        valid_nc = valid.copy()
        valid_nc[[i for i in range(len(topo_names)) if i not in non_complete]] = False
        if valid_nc.sum() >= 2:
            x = lambda2_arr[valid_nc]
            yy = y[valid_nc]
            coeff = np.polyfit(x, yy, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, np.polyval(coeff, xs), ls="--", color=color, alpha=0.55)
    ax.axhline(BETA_C, color="black", ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("lambda_2(L)")
    ax.set_ylabel("beta*")
    ax.set_title("h=0.01 beta* vs graph spectral gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = plt.subplot(3, 2, 6)
    for j, name in enumerate(topo_names):
        if np.isfinite(main.beta_star_p1[j]) and np.isfinite(robust.beta_star_p1[j]):
            ax.scatter(robust.beta_star_p1[j], main.beta_star_p1[j],
                       marker="o", color=colors[j], edgecolor="black", s=70,
                       label=name if j == 0 else None)
        if np.isfinite(main.beta_star_p2[j]) and np.isfinite(robust.beta_star_p2[j]):
            ax.scatter(robust.beta_star_p2[j], main.beta_star_p2[j],
                       marker="s", color=colors[j], edgecolor="black", s=70)
            ax.text(robust.beta_star_p2[j], main.beta_star_p2[j], f" {j+1}", fontsize=8)
    lo, hi = 0.05, 0.55
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("beta* at h=0")
    ax.set_ylabel("beta* at h=0.01")
    ax.set_title("Weak-field shift relative to zero-field exact check")
    ax.grid(alpha=0.3)
    ax.text(0.055, 0.525, "circles: P1; squares: P2", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("Exact decoupling-window scan with weak-field main path and h=0 robustness")
    print("=" * 72)
    print(f"N={N}, beta grid={len(BETA_GRID)} points, beta_c={BETA_C:.6f}")

    topologies = {
        "Grid 4x4": grid_2d_edges(4, periodic=True),
        "Ring": ring_edges(N),
        "Complete": complete_edges(N),
        "Star": star_edges(N),
        "Random Regular (degree 4)": random_regular_edges(N, 4, seed=42),
        "Small World": small_world_edges(N, k=4, p=0.3, seed=42),
    }
    topo_names = list(topologies.keys())
    complete_idx = topo_names.index("Complete")
    non_complete = [i for i in range(len(topo_names)) if i != complete_idx]

    print("\n--- Topologies ---")
    for name, edges in topologies.items():
        degree = np.zeros(N, dtype=int)
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1
        print(
            f"  {name:<16} edges={len(edges):3d} "
            f"degree min/mean/max={degree.min():.0f}/{degree.mean():.2f}/{degree.max():.0f}"
        )

    lambda2s = {name: compute_lambda2(N, edges) for name, edges in topologies.items()}
    lambda2_arr = np.array([lambda2s[name] for name in topo_names])

    print("\n--- Precomputing observables ---")
    obs_cache = {}
    for name, edges in topologies.items():
        obs_cache[name] = precompute_observables(N, edges)
        print(f"  {name:<16} done")

    main_scan = run_scan(H_MAIN, topo_names, obs_cache)
    robust_scan = run_scan(H_ROBUST, topo_names, obs_cache)

    summarize("Main weak-field scan", main_scan, lambda2_arr, topo_names, non_complete)
    summarize("Zero-field robustness scan", robust_scan, lambda2_arr, topo_names, non_complete)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    figure_path = os.path.join(root, "figures", "figure1_exact_decoupling_windows.png")
    csv_path = os.path.join(root, "data", "exact_decoupling_summary.csv")
    npz_path = os.path.join(root, "data", "exact_decoupling_rescan.npz")
    pickle_path = os.path.join(root, "data", "exact_decoupling_rescan.pkl")

    write_summary_csv(csv_path, [main_scan, robust_scan], topo_names, lambda2s)
    save_npz(npz_path, [main_scan, robust_scan], topo_names, lambda2_arr)
    with open(pickle_path, "wb") as f:
        pickle.dump(
            {
                "topologies": topologies,
                "topo_names": topo_names,
                "lambda2s": lambda2s,
                "main": main_scan,
                "robust_h0": robust_scan,
            },
            f,
        )
    make_figure(figure_path, main_scan, robust_scan, topo_names, lambda2_arr, non_complete)

    print("\n--- Written outputs ---")
    print(f"  {figure_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    print(f"  {pickle_path}")


if __name__ == "__main__":
    main()
