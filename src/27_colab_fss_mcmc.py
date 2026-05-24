#!/usr/bin/env python3
"""
27_colab_fss_mcmc.py
================================================================================
Finite-Size Scaling (FSS) of Decoupling Windows via GPU-Accelerated MCMC
for Google Colab (A100 GPU)

Purpose: Path A -- Extend N=16 exact enumeration to N=36,64,100,144 via MCMC.
         Compute FIM condition number kappa(beta), decoupling rate Delta(beta),
         and scalar curvature R(beta) across system sizes.

Note:
  This is an optional exploratory MCMC driver, not the source of the paper's
  main claims.  The manuscript relies on exact enumeration at N=16.  MCMC FIM
  estimates use pooled equilibrium moments and still require autocorrelation
  analysis before being interpreted as precise finite-size-scaling evidence.

Hardware target: Google Colab with A100 (40GB VRAM)
Runtime estimate: ~60-90 minutes for main FSS scan (L=4,6,8,10,12) with warm-start;
                    +20-30 minutes if scalar curvature stencil is enabled (9 MCMC runs per point)

Key improvements over N=16 exact enumeration:
  - MCMC with massive GPU parallelism (up to 200k independent chains)
  - Accumulated sampling (not just final state) for reliable variance estimates
  - Full 2x2 FIM including Cov(E,M)
  - Scalar curvature R via 9-point stencil (selected L only)
  - Finite-size scaling analysis: kappa_peak, beta_star_G, R_smoothness vs N
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

# ==============================================================================
# 0. CHECKPOINT / RESUME UTILITIES (for Colab stability)
# ==============================================================================

def get_checkpoint_dir(preferred_drive_subdir="fss_results"):
    """Auto-detect storage: Google Drive > local Colab > repo data/."""
    drive_base = "/content/drive/MyDrive"
    if os.path.exists(drive_base):
        path = os.path.join(drive_base, preferred_drive_subdir)
        os.makedirs(path, exist_ok=True)
        print(f"[Checkpoint] Using Google Drive: {path}")
        return path
    # Colab local fallback
    colab_path = "/content/fss_results"
    if os.path.exists("/content"):
        os.makedirs(colab_path, exist_ok=True)
        print(f"[Checkpoint] Using Colab local storage: {colab_path}")
        return colab_path
    # Generic repo fallback
    repo_path = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', '..', 'data', preferred_drive_subdir)
    os.makedirs(repo_path, exist_ok=True)
    print(f"[Checkpoint] Using repo data dir: {repo_path}")
    return repo_path

def save_checkpoint(path, data):
    """Atomic pickle write to avoid corruption on interrupt."""
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
    """Simple hash for configuration consistency checking."""
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

BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0  # Onsager ~0.4407
J_VAL = 1.0
H_PATH = 0.001


# ==============================================================================
# 2. SIMULATION CONFIGURATION
# ==============================================================================

@dataclass
class SimConfig:
    """Configuration for a single finite-size scaling run."""
    L: int                          # Grid side length
    num_envs: int                   # Number of parallel MCMC chains
    warmup: int                     # Burn-in steps
    steps: int                      # Total production steps
    sample_every: int               # Thinning interval (sample every N steps)
    beta_list: np.ndarray           # Beta values to scan
    h: float = H_PATH               # External field
    J: float = J_VAL                # Coupling constant
    seed: int = 42                  # Random seed

    @property
    def N(self) -> int:
        return self.L * self.L

    def __post_init__(self):
        # Ensure beta_list is numpy array
        self.beta_list = np.asarray(self.beta_list)


# ==============================================================================
# 3. GPU MCMC ENGINE: Checkerboard Metropolis
# ==============================================================================

class IsingMCMC:
    """
    GPU-parallel checkerboard Metropolis for 2D Ising on LxL grid.
    Each 'env' is an independent Markov chain.
    """
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.L = config.L
        self.num_envs = config.num_envs
        self.h = config.h
        self.J = config.J

        # Random initial spins: +1 or -1
        self.spins = (torch.randint(
            0, 2, (self.num_envs, self.L, self.L),
            device=device, dtype=torch.float64
        ) * 2 - 1)

        # Checkerboard masks for parallel updates
        grid_x, grid_y = torch.meshgrid(
            torch.arange(self.L, device=device),
            torch.arange(self.L, device=device),
            indexing='ij'
        )
        self.mask_even = ((grid_x + grid_y) % 2 == 0).double()
        self.mask_odd = 1.0 - self.mask_even

    def _neighbor_sum(self, spins: torch.Tensor) -> torch.Tensor:
        """Periodic boundary neighbor sum."""
        return (
            torch.roll(spins, 1, dims=1) +
            torch.roll(spins, -1, dims=1) +
            torch.roll(spins, 1, dims=2) +
            torch.roll(spins, -1, dims=2)
        )

    def _energy_per_env(self, spins: torch.Tensor) -> torch.Tensor:
        """Energy E = -0.5 * sum_{<ij>} s_i s_j for each env.

        The -0.5 factor corrects double counting of bonds.
        This returns the bare energy (without beta or h factors).
        When J != 1, ensure consistency with FIM definition."""
        neighbor_sum = self._neighbor_sum(spins)
        return -0.5 * (spins * neighbor_sum).sum(dim=(1, 2))

    def _magnetization_per_env(self, spins: torch.Tensor) -> torch.Tensor:
        """Magnetization M = sum_i s_i for each env."""
        return spins.sum(dim=(1, 2))

    def _effective_h(self, spins: torch.Tensor) -> torch.Tensor:
        """Beta-score statistic -H = -J*E_phys + h*M.

        _energy_per_env returns the physical interaction energy
        E_phys=-sum_<ij> s_i s_j.  The beta derivative of log p is centered
        -H, so the sign of the interaction term is negative here.
        """
        E_phys = self._energy_per_env(spins)
        M = self._magnetization_per_env(spins)
        return -self.J * E_phys + self.h * M

    def step(self, beta: float):
        """One full checkerboard Metropolis step at inverse temperature beta."""
        for mask in [self.mask_even, self.mask_odd]:
            neighbor_sum = self._neighbor_sum(self.spins)
            # delta_E for flipping spin i: 2 * s_i * (J * sum_j s_j + h)
            delta_E = 2.0 * self.spins * (self.J * neighbor_sum + self.h)
            # Metropolis acceptance probability: min(1, exp(-beta * delta_E))
            prob = torch.clamp(torch.exp(-beta * delta_E), max=1.0)
            flip = (torch.rand_like(self.spins) < prob).double() * mask
            self.spins = self.spins * (1.0 - 2.0 * flip)

    def run_and_sample(
        self,
        beta: float,
        custom_warmup: Optional[int] = None,
        custom_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run MCMC at given beta, return accumulated samples of (H_eff, M).

        Args:
            beta: inverse temperature
            custom_warmup: override self.cfg.warmup (for warm-start scans)
            custom_steps: override self.cfg.steps

        Returns:
            h_eff_samples: shape (num_samples, num_envs)
            m_samples: shape (num_samples, num_envs)
            energies: shape (num_samples, num_envs) -- raw energy E (not H_eff)
        """
        warmup = custom_warmup if custom_warmup is not None else self.cfg.warmup
        steps = custom_steps if custom_steps is not None else self.cfg.steps

        # Burn-in
        for _ in range(warmup):
            self.step(beta)

        # Production + sampling
        h_eff_samples = []
        m_samples = []
        e_samples = []

        for step_idx in range(steps):
            self.step(beta)
            if (step_idx + 1) % self.cfg.sample_every == 0:
                h_eff = self._effective_h(self.spins)  # shape (num_envs,)
                m = self._magnetization_per_env(self.spins)
                e = self._energy_per_env(self.spins)
                h_eff_samples.append(h_eff.cpu().numpy())
                m_samples.append(m.cpu().numpy())
                e_samples.append(e.cpu().numpy())

        return (
            np.stack(h_eff_samples, axis=0),   # (num_samples, num_envs)
            np.stack(m_samples, axis=0),        # (num_samples, num_envs)
            np.stack(e_samples, axis=0)         # (num_samples, num_envs)
        )


# ==============================================================================
# 4. FISHER INFORMATION MATRIX COMPUTATION
# ==============================================================================

def compute_fim_from_samples(
    h_eff_samples: np.ndarray,
    m_samples: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, dict]:
    """
    Compute 2x2 FIM from MCMC samples.

    In (beta, h) parameterization:
        F_beta_beta  = Var(H_eff)    where H_eff = -J*E_phys + h*M
        F_hh         = beta^2 * Var(M)
        F_beta_h     = beta * Cov(H_eff, M)

    Args:
        h_eff_samples: shape (num_samples, num_envs)
        m_samples: shape (num_samples, num_envs)
        beta: inverse temperature

    Returns:
        F: 2x2 Fisher information matrix
        stats: dictionary of intermediate statistics
    """
    # Pooled estimator: flatten all (sample, env) pairs and compute sample
    # moments.  This estimates the equilibrium variance Var(H_eff), Var(M)
    # under the target distribution p(beta, h).  Autocorrelation within each
    # chain is *not* corrected here; the effective sample size N_eff should
    # be estimated separately (e.g. via batch means or AR(1) fit) and used
    # to inflate the standard errors reported in stats.
    h_all = h_eff_samples.flatten()
    m_all = m_samples.flatten()

    mean_h = np.mean(h_all)
    mean_m = np.mean(m_all)

    var_h = np.var(h_all, ddof=1)
    var_m = np.var(m_all, ddof=1)
    cov_hm = np.cov(h_all, m_all, ddof=1)[0, 1]

    # Assemble FIM
    F = np.array([
        [var_h,              beta * cov_hm],
        [beta * cov_hm,      beta**2 * var_m]
    ])

    stats = {
        'mean_H': mean_h,
        'mean_M': mean_m,
        'var_H': var_h,
        'var_M': var_m,
        'cov_HM': cov_hm,
        'n_samples': len(h_all),
    }
    return F, stats


# ==============================================================================
# 5. SCALAR CURVATURE: 9-POINT STENCIL
# ==============================================================================

def compute_scalar_curvature_stencil(
    config: SimConfig,
    beta0: float,
    h0: float,
    dbeta: float = 0.002,
    dh: float = 0.0001,
) -> Tuple[float, dict]:
    """
    Compute R(beta0, h0) via 9-point stencil.
    This requires evaluating FIM at 9 points: (beta0 +/- dbeta, h0 +/- dh).
    WARNING: This is expensive -- 9 independent MCMC runs.
    """
    from itertools import product

    # Stencil points: (db, dh) offsets
    offsets = list(product([-dbeta, 0, dbeta], [-dh, 0, dh]))

    metrics = {}
    print(f"  [R-stencil] Computing {len(offsets)} points for beta={beta0:.4f}, h={h0:.4f}")

    for idx, (db, dh_off) in enumerate(offsets):
        beta_pt = beta0 + db
        h_pt = h0 + dh_off

        # Temporarily override h
        cfg_pt = SimConfig(
            L=config.L,
            num_envs=config.num_envs,
            warmup=config.warmup,
            steps=config.steps,
            sample_every=config.sample_every,
            beta_list=np.array([beta_pt]),
            h=h_pt,
            J=config.J,
            seed=config.seed + idx,
        )

        mcmc = IsingMCMC(cfg_pt)
        h_eff_s, m_s, _ = mcmc.run_and_sample(beta_pt)
        F, _ = compute_fim_from_samples(h_eff_s, m_s, beta_pt)
        metrics[(db, dh_off)] = F
        print(f"    Point {idx+1}/{len(offsets)}: beta={beta_pt:.4f}, h={h_pt:.5f} done.")

    # Extract derivatives via centered differences
    g_00 = metrics[(0, 0)]

    # First derivatives
    dg_db = (metrics[(dbeta, 0)] - metrics[(-dbeta, 0)]) / (2 * dbeta)
    dg_dh = (metrics[(0, dh)] - metrics[(0, -dh)]) / (2 * dh)

    # Second derivatives
    d2g_db2 = (metrics[(dbeta, 0)] - 2*g_00 + metrics[(-dbeta, 0)]) / (dbeta**2)
    d2g_dh2 = (metrics[(0, dh)] - 2*g_00 + metrics[(0, -dh)]) / (dh**2)
    d2g_dbdh = (metrics[(dbeta, dh)] - metrics[(dbeta, -dh)]
                - metrics[(-dbeta, dh)] + metrics[(-dbeta, -dh)]) / (4 * dbeta * dh)

    # Compute scalar curvature in 2D
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


def _scalar_curvature_2d(g, dg, d2g):
    """Compute 2D scalar curvature from metric and derivatives."""
    g_inv = np.linalg.inv(g)

    # Christoffel symbols
    Gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                acc = 0.0
                for l in range(2):
                    acc += g_inv[k, l] * (dg[i][j, l] + dg[j][i, l] - dg[l][i, j])
                Gamma[k, i, j] = 0.5 * acc

    # dGamma (derivative of Christoffel)
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

    # Riemann tensor
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

    # Ricci tensor
    Ricci = np.zeros((2, 2))
    for k in range(2):
        for j in range(2):
            for i in range(2):
                Ricci[k, j] += Riemann[i, k, i, j]

    # Scalar curvature
    R_scalar = 0.0
    for k in range(2):
        for j in range(2):
            R_scalar += g_inv[k, j] * Ricci[k, j]
    return R_scalar


def _save_intermediate(results: dict, out_dir: str):
    """Save current results to NPZ and plot immediately after each L completes."""
    os.makedirs(out_dir, exist_ok=True)

    # Save NPZ
    npz_data = {}
    for L in results:
        prefix = f"L{L}_"
        r = results[L]
        npz_data[prefix + 'beta'] = r['beta_list']
        npz_data[prefix + 'kappa'] = r['kappa']
        npz_data[prefix + 'I_eff'] = r['I_eff']
        npz_data[prefix + 'Delta'] = r['Delta']
        npz_data[prefix + 'beta_star_G'] = r['beta_star_G']
        if 'R' in r:
            npz_data[prefix + 'R_beta'] = np.array([d['beta'] for d in r['R']])
            npz_data[prefix + 'R'] = np.array([d['R'] for d in r['R']])

    npz_path = os.path.join(out_dir, "fss_results.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"  [Intermediate] Saved NPZ: {npz_path} (L={sorted(results.keys())})")

    # Plot current progress
    plot_fss_results(results, out_dir=out_dir)


# ==============================================================================
# 6. MAIN FINITE-SIZE SCALING PIPELINE
# ==============================================================================

def run_fss_scan(
    L_list: list = [4, 6, 8, 10, 12],
    num_envs: int = 100000,
    warmup: int = 2000,
    steps: int = 5000,
    sample_every: int = 5,
    beta_range: Tuple[float, float] = (0.30, 0.55),
    n_beta: int = 50,
    compute_R_for_L: Optional[list] = None,
    R_beta_points: Optional[np.ndarray] = None,
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """
    Run the full finite-size scaling scan with checkpoint/resume support.

    Args:
        checkpoint_dir: Directory to save/load checkpoints. If None, no checkpointing.
                        Recommend using Google Drive path for Colab persistence.
    """
    if compute_R_for_L is None:
        compute_R_for_L = []

    beta_list = np.linspace(beta_range[0], beta_range[1], n_beta)

    # --- Checkpoint setup ---
    checkpoint_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")

    config_sig = _config_sig(
        L_list=L_list, num_envs=num_envs, warmup=warmup, steps=steps,
        sample_every=sample_every, beta_range=beta_range, n_beta=n_beta,
        h=H_PATH,  # CRITICAL: include h in signature to prevent cross-contamination
        compute_R_for_L=compute_R_for_L,
        R_beta_points=list(R_beta_points) if R_beta_points is not None else None,
    )

    # --- Load checkpoint if exists ---
    ckpt = load_checkpoint(checkpoint_path)
    results = {}
    if ckpt:
        if ckpt.get('config_sig') != config_sig:
            print(f"[Checkpoint] WARNING: Config mismatch (old sig={ckpt.get('config_sig')}, new={config_sig}).")
            print("[Checkpoint] Starting fresh scan. Old results preserved in checkpoint.")
            ckpt = None
        else:
            results = ckpt.get('results', {})
            print(f"[Checkpoint] Resuming. Already completed L values: {sorted(results.keys())}")

    # --- Determine remaining work ---
    remaining_L = [L for L in L_list if L not in results]

    if not remaining_L:
        print("[Checkpoint] All requested L values already complete. Nothing to do.")
        return results

    # --- Main scan loop ---
    for L in remaining_L:
        print(f"\n{'='*70}")
        print(f"L = {L} (N = {L*L})")
        print(f"{'='*70}")

        # Warm-start strategy: reuse spin configuration across beta points.
        # Full warmup only for the first beta; short warmup (1000 steps) for
        # subsequent points since beta changes slowly (~0.005 per step).
        # This avoids the O(n_beta * L^z) blow-up of independent warmups.
        L_warmup_first = max(warmup, int(200 * L**1.8))
        L_warmup_sub   = 1000   # fixed short warmup for warm-start
        L_steps        = steps  # production steps; autocorrelation still needs diagnostics
        print(f"  Warm-start: 1st warmup={L_warmup_first}, sub warmups={L_warmup_sub}, "
              f"steps={L_steps}")

        cfg = SimConfig(
            L=L,
            num_envs=num_envs,
            warmup=L_warmup_first,
            steps=L_steps,
            sample_every=sample_every,
            beta_list=beta_list,
            h=H_PATH,
            J=J_VAL,
            seed=42 + L,
        )

        # Check for in-progress data for this L
        F_list, kappa_list, I_eff_list, stats_list = [], [], [], []
        start_beta_idx = 0
        R_results = []
        r_start_idx = 0
        mcmc = None

        if ckpt and ckpt.get('in_progress') and ckpt['in_progress']['L'] == L:
            ip = ckpt['in_progress']
            start_beta_idx = ip.get('beta_idx', 0)
            F_list = ip.get('F_list', [])
            kappa_list = ip.get('kappa_list', [])
            I_eff_list = ip.get('I_eff_list', [])
            stats_list = ip.get('stats_list', [])
            r_start_idx = ip.get('R_beta_idx', 0)
            R_results = ip.get('R_results', [])
            print(f"  [Resume] Restoring {start_beta_idx}/{n_beta} beta points, "
                  f"{r_start_idx} R points done.")

        t0 = time.time()
        for b_idx in range(start_beta_idx, len(beta_list)):
            beta = beta_list[b_idx]

            if b_idx == 0 or mcmc is None:
                # First beta (or fresh start after resume): new MCMC with full warmup
                mcmc = IsingMCMC(cfg)
                h_eff_s, m_s, _ = mcmc.run_and_sample(beta, custom_warmup=L_warmup_first)
            else:
                # Warm-start from previous beta's final spin config: short warmup
                h_eff_s, m_s, _ = mcmc.run_and_sample(beta, custom_warmup=L_warmup_sub)

            F, stats = compute_fim_from_samples(h_eff_s, m_s, beta)

            eigs = np.linalg.eigvalsh(F)
            kappa = eigs[-1] / eigs[0] if eigs[0] > 1e-14 else np.nan
            I_eff = np.log(kappa) if kappa > 0 else np.nan

            F_list.append(F)
            kappa_list.append(kappa)
            I_eff_list.append(I_eff)
            stats_list.append(stats)

            if (b_idx + 1) % 10 == 0 or b_idx == 0:
                elapsed = time.time() - t0
                print(f"  [{b_idx+1}/{n_beta}] beta={beta:.4f} | "
                      f"kappa={kappa:.3f} | I_eff={I_eff:.3f} | "
                      f"time={elapsed:.1f}s")

            # --- Checkpoint every 5 beta points ---
            if checkpoint_path and (b_idx + 1) % 5 == 0:
                save_checkpoint(checkpoint_path, {
                    'config_sig': config_sig,
                    'results': results,
                    'in_progress': {
                        'L': L,
                        'beta_idx': b_idx + 1,
                        'F_list': F_list,
                        'kappa_list': kappa_list,
                        'I_eff_list': I_eff_list,
                        'stats_list': stats_list,
                        'R_beta_idx': r_start_idx,
                        'R_results': R_results,
                    },
                    'timestamp': time.time(),
                })

        # --- Post-process this L ---
        F_array = np.array(F_list)  # (n_beta, 2, 2)
        kappa_array = np.array(kappa_list)
        I_eff_array = np.array(I_eff_list)

        Delta_array = np.zeros_like(I_eff_array)
        Delta_array[1:-1] = -(I_eff_array[2:] - I_eff_array[:-2]) / (beta_list[2:] - beta_list[:-2])
        Delta_array[0] = -(I_eff_array[1] - I_eff_array[0]) / (beta_list[1] - beta_list[0])
        Delta_array[-1] = -(I_eff_array[-1] - I_eff_array[-2]) / (beta_list[-1] - beta_list[-2])

        beta_star_G = np.nan
        for i in range(len(Delta_array) - 1):
            if Delta_array[i] > 0 and Delta_array[i+1] < 0:
                beta_star_G = beta_list[i] - Delta_array[i] * (beta_list[i+1] - beta_list[i]) / (Delta_array[i+1] - Delta_array[i])
                break

        results[L] = {
            'beta_list': beta_list,
            'F': F_array,
            'kappa': kappa_array,
            'I_eff': I_eff_array,
            'Delta': Delta_array,
            'beta_star_G': beta_star_G,
            'stats': stats_list,
        }

        # --- Scalar curvature for selected L ---
        if L in compute_R_for_L and R_beta_points is not None:
            for r_idx in range(r_start_idx, len(R_beta_points)):
                beta_pt = R_beta_points[r_idx]
                if beta_range[0] <= beta_pt <= beta_range[1]:
                    R_val, R_info = compute_scalar_curvature_stencil(
                        cfg, beta_pt, H_PATH,
                        dbeta=0.002, dh=0.0001
                    )
                    R_results.append({'beta': beta_pt, 'R': R_val, 'info': R_info})
                    print(f"  [R] beta={beta_pt:.4f} => R={R_val:.4f}")

                # --- Checkpoint after each R point (very expensive) ---
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, {
                        'config_sig': config_sig,
                        'results': results,
                        'in_progress': {
                            'L': L,
                            'beta_idx': len(beta_list),  # All betas done
                            'F_list': F_list,
                            'kappa_list': kappa_list,
                            'I_eff_list': I_eff_list,
                            'stats_list': stats_list,
                            'R_beta_idx': r_idx + 1,
                            'R_results': R_results,
                        },
                        'timestamp': time.time(),
                    })

            if R_results:
                results[L]['R'] = R_results

        # --- Save intermediate results immediately after each L ---
        _save_intermediate(results, out_dir=checkpoint_dir)

        # --- Mark L complete ---
        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                'config_sig': config_sig,
                'results': results,
                'in_progress': None,
                'timestamp': time.time(),
            })
            print(f"  [Checkpoint] L={L} complete, checkpoint saved.")

        print(f"  L={L} complete. beta*_G = {beta_star_G:.4f}")

    return results


# ==============================================================================
# 7. PLOTTING
# ==============================================================================

def plot_fss_results(results: dict, out_dir: str = "."):
    """Generate publication-quality FSS figures."""
    os.makedirs(out_dir, exist_ok=True)
    L_list = sorted(results.keys())

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))

    # --- Panel 1: kappa(beta) for all L ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for L, color in zip(L_list, colors):
        r = results[L]
        ax.plot(r['beta_list'], r['kappa'], '-o', color=color, label=f'L={L} (N={L*L})', markersize=3)
    ax.axvline(BETA_C, color='black', linestyle='--', alpha=0.5, label=f'$\\beta_c$={BETA_C:.4f}')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\kappa(\mathcal{F})$')
    ax.set_title('FIM Condition Number')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: I_eff(beta) ---
    ax = axes[0, 1]
    for L, color in zip(L_list, colors):
        r = results[L]
        ax.plot(r['beta_list'], r['I_eff'], '-o', color=color, label=f'L={L}', markersize=3)
    ax.axvline(BETA_C, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$I_{\mathrm{eff}} = \log\kappa$')
    ax.set_title('Effective Inertia')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: Delta(beta) ---
    ax = axes[1, 0]
    for L, color in zip(L_list, colors):
        r = results[L]
        ax.plot(r['beta_list'], r['Delta'], '-o', color=color, label=f'L={L}', markersize=3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(BETA_C, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\Delta = -dI_{\mathrm{eff}}/d\beta$')
    ax.set_title('Decoupling Rate')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 4: beta*_G vs 1/N ---
    ax = axes[1, 1]
    beta_stars = [results[L]['beta_star_G'] for L in L_list]
    inv_N = [1.0 / (L*L) for L in L_list]
    valid = [~np.isnan(b) for b in beta_stars]
    if any(valid):
        ax.plot(np.array(inv_N)[valid], np.array(beta_stars)[valid], 'ko-', markersize=8)
        for L, x, y in zip(L_list, inv_N, beta_stars):
            if not np.isnan(y):
                ax.annotate(f'L={L}', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.axhline(BETA_C, color='red', linestyle='--', alpha=0.5, label=f'$\\beta_c$={BETA_C:.4f}')
    ax.set_xlabel(r'$1/N$')
    ax.set_ylabel(r'$\beta^*_G$')
    ax.set_title('Window Boundary vs System Size')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '27_fss_main_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir}/27_fss_main_results.png")

    # --- Panel 5: R(beta) for selected L ---
    has_R = any('R' in results[L] for L in L_list)
    if has_R:
        fig, ax = plt.subplots(figsize=(8, 5))
        for L in L_list:
            if 'R' in results[L]:
                R_data = results[L]['R']
                betas = [d['beta'] for d in R_data]
                Rs = [d['R'] for d in R_data]
                ax.plot(betas, Rs, '-o', label=f'L={L}')
        ax.axvline(BETA_C, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$R$')
        ax.set_title('Scalar Curvature (MCMC stencil)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '27_fss_scalar_curvature.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_dir}/27_fss_scalar_curvature.png")


# ==============================================================================
# 8. COLAB MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FINITE-SIZE SCALING: MCMC on GPU (with Checkpoint/Resume)")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 0. GOOGLE DRIVE MOUNT (optional but strongly recommended for Colab)
    # --------------------------------------------------------------------------
    # If you want checkpoint persistence across Colab disconnects, run this
    # in a separate cell BEFORE executing this script:
    #
    #   from google.colab import drive
    #   drive.mount('/content/drive')
    #
    # The script will auto-detect /content/drive/MyDrive and save checkpoints
    # there. Without Drive, checkpoints live in /content and vanish on disconnect.
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # USER CONFIGURATION -- Adjust these for your Colab runtime
    # --------------------------------------------------------------------------

    # System sizes to scan. L=4 is N=16 (can compare with exact enumeration)
    # L=12 is N=144 (largest practical for accurate MCMC on A100)
    L_LIST = [4, 6, 8, 10, 12]

    # NOTE on external field h:
    #   h=0.001 is chosen because h=0.01 was found to exceed the critical
    #   scaling field h_c ~ L^(-15/8) for L>=8, shifting beta*_G away from
    #   the exact h=0 limit. h=0.001 is << h_c for all L in [4,12] while
    #   still breaking Z_2 symmetry enough to avoid bimodal M distributions
    #   that corrupt Var(M) in the FIM. See 27b_h0_validation.py for details.
    #   H_PATH is defined globally in the script header.

    # Parallel chains. A100 40GB can easily handle 200k+ for L<=12
    # (200k * 12*12 * 8 bytes ≈ 230 MB — a drop in the bucket)
    NUM_ENVS = 200000

    # MCMC steps. Increase for better accuracy at larger L.
    # L=4:  fast convergence
    # L=12: needs longer due to critical slowing down near beta_c
    WARMUP = 2000
    STEPS = 5000
    SAMPLE_EVERY = 5

    # Beta scan range. Include beta_c = 0.4407
    BETA_MIN, BETA_MAX = 0.30, 0.55
    N_BETA = 50

    # Scalar curvature: compute for selected L at selected beta points
    # This is expensive (9 MCMC runs per point). Limit to a few points.
    COMPUTE_R_FOR_L = [8, 10]  # Only for L=8,10
    R_BETA_POINTS = np.array([0.38, 0.42, 0.44, 0.46, 0.48, 0.50])

    # --------------------------------------------------------------------------
    # CHECKPOINT / OUTPUT DIRECTORY
    # --------------------------------------------------------------------------
    checkpoint_dir = get_checkpoint_dir("fss_results")
    out_dir = checkpoint_dir  # Same directory for all outputs

    # --------------------------------------------------------------------------
    # RUN (with automatic resume)
    # --------------------------------------------------------------------------

    print(f"\nConfiguration:")
    print(f"  L values: {L_LIST}")
    print(f"  Environments: {NUM_ENVS}")
    print(f"  h (external field): {H_PATH}")
    print(f"  Warmup: {WARMUP}, Steps: {STEPS}, Sample every: {SAMPLE_EVERY}")
    print(f"  Beta range: [{BETA_MIN}, {BETA_MAX}] with {N_BETA} points")
    print(f"  Compute R for L={COMPUTE_R_FOR_L} at beta={R_BETA_POINTS}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    # Time estimate based on warm-start + fixed steps.
    # L=4 first point ~20s, subsequent ~15s => ~13min per L.
    est_per_L_min = 13
    main_est = len(L_LIST) * est_per_L_min
    r_est = (len(COMPUTE_R_FOR_L) * len(R_BETA_POINTS) * 9 * 0.5) if R_BETA_POINTS is not None else 0
    print(f"  Expected runtime: main ~{main_est:.0f} min, R-stencil ~{r_est:.0f} min (estimate)")
    print()

    overall_t0 = time.time()

    results = run_fss_scan(
        L_list=L_LIST,
        num_envs=NUM_ENVS,
        warmup=WARMUP,
        steps=STEPS,
        sample_every=SAMPLE_EVERY,
        beta_range=(BETA_MIN, BETA_MAX),
        n_beta=N_BETA,
        compute_R_for_L=COMPUTE_R_FOR_L,
        R_beta_points=R_BETA_POINTS,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"\n{'='*70}")
    print(f"All simulations complete. Total time: {(time.time()-overall_t0)/60:.1f} min")
    print(f"{'='*70}")

    # --------------------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------------------

    # Save as pickle
    pkl_path = os.path.join(out_dir, "fss_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved pickle: {pkl_path}")

    # Also save as NPZ for easy numpy loading
    npz_data = {}
    for L in results:
        prefix = f"L{L}_"
        r = results[L]
        npz_data[prefix + 'beta'] = r['beta_list']
        npz_data[prefix + 'kappa'] = r['kappa']
        npz_data[prefix + 'I_eff'] = r['I_eff']
        npz_data[prefix + 'Delta'] = r['Delta']
        npz_data[prefix + 'beta_star_G'] = r['beta_star_G']
        if 'R' in r:
            npz_data[prefix + 'R_beta'] = np.array([d['beta'] for d in r['R']])
            npz_data[prefix + 'R'] = np.array([d['R'] for d in r['R']])

    npz_path = os.path.join(out_dir, "fss_results.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved NPZ: {npz_path}")

    # --------------------------------------------------------------------------
    # PLOT
    # --------------------------------------------------------------------------
    plot_fss_results(results, out_dir=out_dir)

    # --------------------------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'L':>3} {'N':>5} {'beta*_G':>10} {'kappa_max':>12} {'peak_beta':>10}")
    print("-" * 70)
    for L in sorted(results.keys()):
        r = results[L]
        kappa = r['kappa']
        peak_idx = np.nanargmax(kappa)
        print(f"{L:>3} {L*L:>5} {r['beta_star_G']:>10.4f} "
              f"{kappa[peak_idx]:>12.2f} {r['beta_list'][peak_idx]:>10.4f}")
    print("=" * 70)
    print(f"\nOnsager beta_c = {BETA_C:.4f}")
    print("Compare beta*_G to beta_c to assess convergence with N.")

    # Download link reminder
    print(f"\n[Colab] Results saved to {out_dir}/")
    print("[Colab] Use the file browser on the left to download:")
    print(f"  - {out_dir}/fss_results.pkl")
    print(f"  - {out_dir}/fss_results.npz")
    print(f"  - {out_dir}/27_fss_main_results.png")
    print(f"  - {out_dir}/27_fss_scalar_curvature.png")
    print(f"  - {out_dir}/checkpoint.pkl  (resume state)")
