#!/usr/bin/env python3
"""
28_R_fss_mcmc.py
================================================================================
Finite-Size Scaling of Scalar Curvature R(beta) via GPU-Accelerated MCMC
for Google Colab (A100 GPU)

Purpose: Path A (redirected) -- Compute coordinate-invariant scalar curvature
R(beta) for L=4,6,8,10,12 using full 2D stencil at fixed h=0.01.

Key design choices:
  - Full 2D stencil (not beta-only): R requires Christoffel symbols from
    derivatives in BOTH beta and h directions.
  - Fixed h=0.01: safely away from Z2 singularity at h=0.
  - delta_beta = 0.002, delta_h = 0.002: larger than exact enumeration to
    avoid MCMC noise drowning the finite-difference signal.
  - Warm-start between adjacent beta points to control runtime.
  - Checkpoint every beta point (since each point is 9 independent MCMC runs).

Expected runtime: ~2-3 hours for L=4,6,8,10,12 (A100, 200k chains).
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import hashlib
from dataclasses import dataclass
from typing import Tuple, Optional
from itertools import product

# ==============================================================================
# 0. CHECKPOINT / RESUME UTILITIES
# ==============================================================================

def get_checkpoint_dir(preferred_drive_subdir="fss_results_R"):
    """Auto-detect storage: Google Drive > local Colab > repo data/."""
    drive_base = "/content/drive/MyDrive"
    if os.path.exists(drive_base):
        path = os.path.join(drive_base, preferred_drive_subdir)
        os.makedirs(path, exist_ok=True)
        print(f"[Checkpoint] Using Google Drive: {path}")
        return path
    colab_path = "/content/fss_results_R"
    if os.path.exists("/content"):
        os.makedirs(colab_path, exist_ok=True)
        print(f"[Checkpoint] Using Colab local storage: {colab_path}")
        return colab_path
    repo_path = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', '..', 'data', preferred_drive_subdir)
    os.makedirs(repo_path, exist_ok=True)
    print(f"[Checkpoint] Using repo data dir: {repo_path}")
    return repo_path

def save_checkpoint(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp, path)

def load_checkpoint(path):
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def _config_sig(**kwargs):
    return hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:16]


# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 70)
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 70)

BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0  # Onsager exact: ~0.4407
J_VAL = 1.0
H_CENTER = 0.01        # Fixed h for stencil center
DELTA_BETA = 0.002     # Stencil step in beta
DELTA_H = 0.002        # Stencil step in h (MCMC-robust, larger than exact enum)


# ==============================================================================
# 2. SIMULATION CONFIGURATION
# ==============================================================================

@dataclass
class SimConfig:
    L: int
    num_envs: int
    warmup: int
    steps: int
    sample_every: int
    beta_list: np.ndarray
    h: float = H_CENTER
    J: float = J_VAL
    seed: int = 42

    @property
    def N(self) -> int:
        return self.L * self.L

    def __post_init__(self):
        self.beta_list = np.asarray(self.beta_list)


# ==============================================================================
# 3. GPU MCMC ENGINE
# ==============================================================================

class IsingMCMC:
    def __init__(self, config: SimConfig, initial_spins: Optional[torch.Tensor] = None):
        self.cfg = config
        self.L = config.L
        self.num_envs = config.num_envs
        self.h = config.h
        self.J = config.J

        if initial_spins is not None:
            self.spins = initial_spins.clone()
        else:
            self.spins = (torch.randint(
                0, 2, (self.num_envs, self.L, self.L),
                device=device, dtype=torch.float64
            ) * 2 - 1)

        grid_x, grid_y = torch.meshgrid(
            torch.arange(self.L, device=device),
            torch.arange(self.L, device=device),
            indexing='ij'
        )
        self.mask_even = ((grid_x + grid_y) % 2 == 0).double()
        self.mask_odd = 1.0 - self.mask_even

    def _neighbor_sum(self, spins):
        return (
            torch.roll(spins, 1, dims=1) +
            torch.roll(spins, -1, dims=1) +
            torch.roll(spins, 1, dims=2) +
            torch.roll(spins, -1, dims=2)
        )

    def _energy_per_env(self, spins):
        return -0.5 * (spins * self._neighbor_sum(spins)).sum(dim=(1, 2))

    def _magnetization_per_env(self, spins):
        return spins.sum(dim=(1, 2))

    def _effective_h(self, spins):
        E_phys = self._energy_per_env(spins)
        M = self._magnetization_per_env(spins)
        return -self.J * E_phys + self.h * M

    def step(self, beta: float):
        for mask in [self.mask_even, self.mask_odd]:
            neighbor_sum = self._neighbor_sum(self.spins)
            delta_E = 2.0 * self.spins * (self.J * neighbor_sum + self.h)
            prob = torch.clamp(torch.exp(-beta * delta_E), max=1.0)
            flip = (torch.rand_like(self.spins) < prob).double() * mask
            self.spins = self.spins * (1.0 - 2.0 * flip)

    def run_and_sample(self, beta: float, warmup: int, steps: int):
        for _ in range(warmup):
            self.step(beta)

        h_eff_samples = []
        m_samples = []
        for step_idx in range(steps):
            self.step(beta)
            if (step_idx + 1) % self.cfg.sample_every == 0:
                h_eff = self._effective_h(self.spins)
                m = self._magnetization_per_env(self.spins)
                h_eff_samples.append(h_eff.cpu().numpy())
                m_samples.append(m.cpu().numpy())

        return np.stack(h_eff_samples, axis=0), np.stack(m_samples, axis=0)


# ==============================================================================
# 4. FIM COMPUTATION (batch means)
# ==============================================================================

def compute_fim_from_samples(h_eff_samples, m_samples, beta):
    """
    Estimate FIM from MCMC samples by pooling all (sample, env) pairs.

    NOTE: h_eff = -J*E_phys + h*M is the beta-score statistic before centering,
    where E_phys=-sum_<ij> s_i s_j is the physical interaction energy returned
    by _energy_per_env.

    NOTE on estimator: we pool all samples and compute sample moments.
    Autocorrelation within chains is NOT corrected here; N_eff should be
    estimated separately for reliable error bars.
    """
    h_all = h_eff_samples.flatten()
    m_all = m_samples.flatten()

    var_h = np.var(h_all, ddof=1)
    var_m = np.var(m_all, ddof=1)
    cov_hm = np.cov(h_all, m_all, ddof=1)[0, 1]

    F = np.array([
        [var_h,              beta * cov_hm],
        [beta * cov_hm,      beta**2 * var_m]
    ])
    return F


# ==============================================================================
# 5. SCALAR CURVATURE (2D)
# ==============================================================================

def _scalar_curvature_2d(g, dg, d2g):
    g_inv = np.linalg.inv(g)

    Gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                acc = 0.0
                for l in range(2):
                    acc += g_inv[k, l] * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                Gamma[k, i, j] = 0.5 * acc

    dGamma = np.zeros((2, 2, 2, 2))
    for m in range(2):
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    acc = 0.0
                    for l in range(2):
                        dginv_kl = 0.0
                        for a in range(2):
                            for b in range(2):
                                dginv_kl -= g_inv[k, a] * g_inv[l, b] * dg[m][a, b]
                        acc += dginv_kl * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                        acc += g_inv[k, l] * (
                            d2g[m][i][j, l] + d2g[m][j][i, l] - d2g[m][l][i, j]
                        )
                    dGamma[m, k, i, j] = 0.5 * acc

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

    Ricci = np.zeros((2, 2))
    for k in range(2):
        for j in range(2):
            for i in range(2):
                Ricci[k, j] += Riemann[i, k, i, j]

    R_scalar = 0.0
    for k in range(2):
        for j in range(2):
            R_scalar += g_inv[k, j] * Ricci[k, j]
    return R_scalar


def compute_R_at_beta(
    cfg: SimConfig,
    beta0: float,
    h0: float,
    dbeta: float = DELTA_BETA,
    dh: float = DELTA_H,
) -> Tuple[float, dict]:
    """
    Compute R(beta0, h0) via 9-point stencil with warm-start between points.
    Returns scalar curvature and diagnostic info.
    """
    offsets = list(product([-dbeta, 0, dbeta], [-dh, 0, dh]))
    metrics = {}
    print(f"  [R-stencil] beta={beta0:.4f}, h={h0:.4f} | {len(offsets)} points")

    # Adaptive warmup: critical slowing down near beta_c
    beta_dist = abs(beta0 - BETA_C)
    if beta_dist < 0.03:
        adaptive_factor = 4.0
    elif beta_dist < 0.06:
        adaptive_factor = 2.0
    else:
        adaptive_factor = 1.0

    # Warm-start: first point gets full (adaptive) warmup; subsequent points
    # reuse the final spin config from the previous point and get a short
    # thermalization (20% of full) since they are nearby in parameter space.
    first_warmup = int(cfg.warmup * adaptive_factor)
    sub_warmup = max(500, int(first_warmup * 0.2))

    prev_spins = None

    # Run MCMC at each stencil point
    for idx, (db, dh_off) in enumerate(offsets):
        beta_pt = beta0 + db
        h_pt = h0 + dh_off

        cfg_pt = SimConfig(
            L=cfg.L,
            num_envs=cfg.num_envs,
            warmup=first_warmup if idx == 0 else sub_warmup,
            steps=cfg.steps,
            sample_every=cfg.sample_every,
            beta_list=np.array([beta_pt]),
            h=h_pt,
            J=cfg.J,
            seed=cfg.seed + idx,
        )

        mcmc = IsingMCMC(cfg_pt, initial_spins=prev_spins)
        h_eff_s, m_s = mcmc.run_and_sample(beta_pt, cfg_pt.warmup, cfg_pt.steps)
        F = compute_fim_from_samples(h_eff_s, m_s, beta_pt)
        metrics[(db, dh_off)] = F
        prev_spins = mcmc.spins  # warm-start for next point
        print(f"    Point {idx+1}/{len(offsets)}: beta={beta_pt:.4f}, h={h_pt:.5f} "
              f"(warmup={cfg_pt.warmup}) done.")

    # Extract derivatives
    g_00 = metrics[(0, 0)]

    dg_db = (metrics[(dbeta, 0)] - metrics[(-dbeta, 0)]) / (2 * dbeta)
    dg_dh = (metrics[(0, dh)] - metrics[(0, -dh)]) / (2 * dh)

    d2g_db2 = (metrics[(dbeta, 0)] - 2*g_00 + metrics[(-dbeta, 0)]) / (dbeta**2)
    d2g_dh2 = (metrics[(0, dh)] - 2*g_00 + metrics[(0, -dh)]) / (dh**2)
    d2g_dbdh = (metrics[(dbeta, dh)] - metrics[(dbeta, -dh)]
                - metrics[(-dbeta, dh)] + metrics[(-dbeta, -dh)]) / (4 * dbeta * dh)

    R = _scalar_curvature_2d(
        g_00,
        [dg_db, dg_dh],
        [[d2g_db2, d2g_dbdh], [d2g_dbdh, d2g_dh2]]
    )

    info = {
        'g_00': g_00,
        'dg_db': dg_db,
        'dg_dh': dg_dh,
        'd2g_db2': d2g_db2,
        'd2g_dh2': d2g_dh2,
        'd2g_dbdh': d2g_dbdh,
    }
    return R, info


# ==============================================================================
# 6. MAIN R-FSS PIPELINE
# ==============================================================================

def run_R_fss_scan(
    L_list: list = [4, 6, 8, 10, 12],
    num_envs: int = 200000,
    warmup: int = 2000,
    steps: int = 5000,
    sample_every: int = 5,
    beta_range: Tuple[float, float] = (0.38, 0.50),
    n_beta: int = 15,
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """
    Run scalar curvature R(beta) finite-size scaling scan.
    """
    beta_list = np.linspace(beta_range[0], beta_range[1], n_beta)

    checkpoint_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_R.pkl")

    config_sig = _config_sig(
        L_list=L_list, num_envs=num_envs, warmup=warmup, steps=steps,
        sample_every=sample_every, beta_range=beta_range, n_beta=n_beta,
        h=H_CENTER, dbeta=DELTA_BETA, dh=DELTA_H,
    )

    ckpt = load_checkpoint(checkpoint_path)
    results = {}
    if ckpt:
        if ckpt.get('config_sig') != config_sig:
            print(f"[Checkpoint] WARNING: Config mismatch. Starting fresh.")
            ckpt = None
        else:
            results = ckpt.get('results', {})
            print(f"[Checkpoint] Resuming. Completed L: {sorted(results.keys())}")

    remaining_L = [L for L in L_list if L not in results]
    if not remaining_L:
        print("[Checkpoint] All L complete. Nothing to do.")
        return results

    for L in remaining_L:
        print(f"\n{'='*70}")
        print(f"L = {L} (N = {L*L}) | R(beta) scan")
        print(f"{'='*70}")

        # Adaptive warmup for critical slowing down
        L_warmup = max(warmup, int(200 * L**1.8))
        L_steps = steps
        print(f"  Warmup={L_warmup}, Steps={L_steps}, Sample every={sample_every}")

        cfg = SimConfig(
            L=L, num_envs=num_envs, warmup=L_warmup, steps=L_steps,
            sample_every=sample_every, beta_list=beta_list,
            h=H_CENTER, J=J_VAL, seed=42 + L,
        )

        R_list = []
        info_list = []
        t0 = time.time()

        for b_idx, beta in enumerate(beta_list):
            R_val, R_info = compute_R_at_beta(
                cfg, beta, H_CENTER,
                dbeta=DELTA_BETA, dh=DELTA_H
            )
            R_list.append(R_val)
            info_list.append(R_info)

            elapsed = time.time() - t0
            print(f"  [{b_idx+1}/{n_beta}] beta={beta:.4f} | R={R_val:12.4f} | time={elapsed/60:.1f}min")

            # Checkpoint after every beta point (since each is expensive)
            if checkpoint_path:
                save_checkpoint(checkpoint_path, {
                    'config_sig': config_sig,
                    'results': {**results, L: {
                        'beta_list': beta_list[:b_idx+1],
                        'R': np.array(R_list),
                        'info': info_list,
                    }},
                    'timestamp': time.time(),
                })

        results[L] = {
            'beta_list': beta_list,
            'R': np.array(R_list),
            'info': info_list,
        }

        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                'config_sig': config_sig,
                'results': results,
                'timestamp': time.time(),
            })
            print(f"  [Checkpoint] L={L} complete.")

    return results


# ==============================================================================
# 7. PLOTTING
# ==============================================================================

def plot_R_results(results: dict, out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    L_list = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for L, color in zip(L_list, colors):
        r = results[L]
        ax.plot(r['beta_list'], r['R'], '-o', color=color,
                label=f'L={L} (N={L*L})', markersize=5)

    ax.axvline(BETA_C, color='black', linestyle='--', alpha=0.5,
               label=f'$\\beta_c$={BETA_C:.4f}')
    ax.set_xlabel(r'$\\beta$')
    ax.set_ylabel(r'$R(\\beta)$')
    ax.set_title('Scalar Curvature $R(\\beta)$: Finite-Size Scaling')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '28_R_fss_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir}/28_R_fss_results.png")


# ==============================================================================
# 8. COLAB MAIN
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SCALAR CURVATURE R(beta): FSS via MCMC")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    L_LIST = [4, 6, 8, 10, 12]
    NUM_ENVS = 200000
    WARMUP = 2000
    STEPS = 5000
    SAMPLE_EVERY = 5

    # Beta range centered on beta_c = 0.4407
    BETA_MIN, BETA_MAX = 0.38, 0.50
    N_BETA = 15  # 15 points = 135 MCMC runs per L

    # --------------------------------------------------------------------------
    # CHECKPOINT / OUTPUT
    # --------------------------------------------------------------------------
    checkpoint_dir = get_checkpoint_dir("fss_results_R")
    out_dir = checkpoint_dir

    print(f"\nConfiguration:")
    print(f"  L values: {L_LIST}")
    print(f"  Environments: {NUM_ENVS}")
    print(f"  h (fixed): {H_CENTER}")
    print(f"  Stencil: delta_beta={DELTA_BETA}, delta_h={DELTA_H}")
    print(f"  Beta range: [{BETA_MIN}, {BETA_MAX}] with {N_BETA} points")
    print(f"  Expected: ~{len(L_LIST) * N_BETA * 9 * 0.4:.0f} min total")
    print()

    overall_t0 = time.time()

    results = run_R_fss_scan(
        L_list=L_LIST,
        num_envs=NUM_ENVS,
        warmup=WARMUP,
        steps=STEPS,
        sample_every=SAMPLE_EVERY,
        beta_range=(BETA_MIN, BETA_MAX),
        n_beta=N_BETA,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"\n{'='*70}")
    print(f"Complete. Total time: {(time.time()-overall_t0)/60:.1f} min")
    print(f"{'='*70}")

    # Save
    pkl_path = os.path.join(out_dir, "R_fss_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle: {pkl_path}")

    npz_data = {}
    for L in results:
        prefix = f"L{L}_"
        r = results[L]
        npz_data[prefix + 'beta'] = r['beta_list']
        npz_data[prefix + 'R'] = r['R']

    npz_path = os.path.join(out_dir, "R_fss_results.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved NPZ: {npz_path}")

    # Plot
    plot_R_results(results, out_dir=out_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: R(beta) near beta_c")
    print("=" * 70)
    print(f"{'L':>3} {'N':>5} {'R(beta_c)':>12} {'R_min':>12} {'R_max':>12}")
    print("-" * 70)
    for L in sorted(results.keys()):
        r = results[L]
        R = r['R']
        R_at_c = np.interp(BETA_C, r['beta_list'], R)
        print(f"{L:>3} {L*L:>5} {R_at_c:>12.2f} {np.nanmin(R):>12.2f} {np.nanmax(R):>12.2f}")
    print("=" * 70)
    print(f"\nJanke-Johnston prediction: R ~ |beta - beta_c|^(-2) diverges at beta_c")
    print("Look for increasing |R| with L near beta_c.")

    print(f"\n[Colab] Results in {out_dir}/")
    print("  - R_fss_results.pkl / .npz")
    print("  - 28_R_fss_results.png")
    print("  - checkpoint_R.pkl")
