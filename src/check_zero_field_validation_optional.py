#!/usr/bin/env python3
"""
check_zero_field_validation_optional.py
================================================================================
Quick validation: L=4, h=0 MCMC vs exact enumeration (N=16).

Purpose: Verify that warm-start GPU MCMC with a pooled-moment FIM estimator
approximately reproduces the exact-enumeration result beta*_G ~ 0.476.

Large deviations from the exact value indicate MCMC thermalization,
autocorrelation, or FIM-estimator issues.

Runtime: ~10-12 minutes on A100 (L=4, 50 beta points, 200k chains).
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple

# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 70)
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0  # Onsager ~0.4407
J_VAL = 1.0
H_VAL = 0.0  # KEY: zero external field

# ==============================================================================
# 2. GPU MCMC ENGINE (minimal, self-contained)
# ==============================================================================

class IsingMCMC:
    """Minimal GPU checkerboard Metropolis for LxL Ising, h=0."""
    def __init__(self, L: int, num_envs: int):
        self.L = L
        self.num_envs = num_envs
        self.spins = (torch.randint(
            0, 2, (num_envs, L, L), device=device, dtype=torch.float64
        ) * 2 - 1)

        grid_x, grid_y = torch.meshgrid(
            torch.arange(L, device=device),
            torch.arange(L, device=device),
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

    def step(self, beta: float):
        for mask in [self.mask_even, self.mask_odd]:
            neighbor_sum = self._neighbor_sum(self.spins)
            delta_E = 2.0 * self.spins * (J_VAL * neighbor_sum)
            prob = torch.clamp(torch.exp(-beta * delta_E), max=1.0)
            flip = (torch.rand_like(self.spins) < prob).double() * mask
            self.spins = self.spins * (1.0 - 2.0 * flip)

    def run_and_sample(self, beta: float, warmup: int, steps: int, sample_every: int):
        for _ in range(warmup):
            self.step(beta)

        e_samples = []
        m_samples = []
        for step_idx in range(steps):
            self.step(beta)
            if (step_idx + 1) % sample_every == 0:
                e_samples.append(self._energy_per_env(self.spins).cpu().numpy())
                m_samples.append(self._magnetization_per_env(self.spins).cpu().numpy())

        return np.stack(e_samples, axis=0), np.stack(m_samples, axis=0)


# ==============================================================================
# 3. FIM COMPUTATION (pooled equilibrium moments)
# ==============================================================================

def compute_fim(e_samples, m_samples, beta):
    """
    Compute 2x2 FIM from MCMC samples in (beta, h) parameterization.
    _energy_per_env returns the physical interaction energy E_phys=-sum s_i s_j.
    The beta score statistic is -H = -J*E_phys + h*M.  At h=0 the sign does not
    affect Var(score), but it does matter for Cov(score,M) away from h=0.
    """
    score = -J_VAL * e_samples + H_VAL * m_samples

    score_all = score.reshape(-1)
    m_all = m_samples.reshape(-1)

    var_score = np.var(score_all, ddof=1)
    var_m = np.var(m_all, ddof=1)
    cov_score_m = np.cov(score_all, m_all, ddof=1)[0, 1]

    F = np.array([
        [var_score,              beta * cov_score_m],
        [beta * cov_score_m,     beta**2 * var_m]
    ])
    return F


# ==============================================================================
# 4. MAIN SCAN
# ==============================================================================

def main():
    L = 4
    NUM_ENVS = 200000
    WARMUP_FIRST = 5000   # Full warmup for first beta
    WARMUP_SUB = 1000     # Warm-start for subsequent betas
    STEPS = 5000
    SAMPLE_EVERY = 5
    BETA_LIST = np.linspace(0.30, 0.55, 50)

    print(f"\nValidation Scan: L={L}, h={H_VAL}, {NUM_ENVS} chains, {len(BETA_LIST)} beta points")
    print("Expected: beta*_G near 0.476 (exact enumeration, N=16, h=0)")
    print("Large deviations indicate MCMC thermalization/autocorrelation or FIM-estimator issues.\n")

    mcmc = None
    F_list = []
    kappa_list = []
    I_eff_list = []

    t0 = time.time()
    for b_idx, beta in enumerate(BETA_LIST):
        if b_idx == 0 or mcmc is None:
            mcmc = IsingMCMC(L, NUM_ENVS)
            e_s, m_s = mcmc.run_and_sample(beta, WARMUP_FIRST, STEPS, SAMPLE_EVERY)
        else:
            e_s, m_s = mcmc.run_and_sample(beta, WARMUP_SUB, STEPS, SAMPLE_EVERY)

        F = compute_fim(e_s, m_s, beta)
        eigs = np.linalg.eigvalsh(F)
        kappa = eigs[-1] / eigs[0] if eigs[0] > 1e-14 else np.nan
        I_eff = np.log(kappa) if kappa > 0 else np.nan

        F_list.append(F)
        kappa_list.append(kappa)
        I_eff_list.append(I_eff)

        if (b_idx + 1) % 10 == 0 or b_idx == 0:
            print(f"  [{b_idx+1:2d}/{len(BETA_LIST)}] beta={beta:.4f} | "
                  f"kappa={kappa:8.2f} | I_eff={I_eff:.3f} | "
                  f"time={(time.time()-t0)/60:.1f}min")

    kappa = np.array(kappa_list)
    I_eff = np.array(I_eff_list)

    # Delta = -dI_eff/dbeta
    Delta = np.zeros_like(I_eff)
    Delta[1:-1] = -(I_eff[2:] - I_eff[:-2]) / (BETA_LIST[2:] - BETA_LIST[:-2])
    Delta[0] = -(I_eff[1] - I_eff[0]) / (BETA_LIST[1] - BETA_LIST[0])
    Delta[-1] = -(I_eff[-1] - I_eff[-2]) / (BETA_LIST[-1] - BETA_LIST[-2])

    # Find zero crossing
    beta_star_G = np.nan
    for i in range(len(Delta) - 1):
        if Delta[i] > 0 and Delta[i+1] < 0:
            beta_star_G = BETA_LIST[i] - Delta[i] * (BETA_LIST[i+1] - BETA_LIST[i]) / (Delta[i+1] - Delta[i])
            break

    peak_idx = np.nanargmax(kappa)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"beta*_G (MCMC, h=0)     = {beta_star_G:.4f}")
    print("beta*_G (exact, N=16)   ~ 0.476  (exact enumeration)")
    print(f"Onsager beta_c          = {BETA_C:.4f}")
    print(f"kappa_max               = {kappa[peak_idx]:.1f} @ beta={BETA_LIST[peak_idx]:.4f}")
    print(f"Total time              = {(time.time()-t0)/60:.1f} min")
    print(f"{'='*70}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(BETA_LIST, kappa, 'b.-')
    ax.axvline(BETA_C, color='r', ls='--', alpha=0.5, label=f'beta_c={BETA_C:.4f}')
    ax.axvline(beta_star_G, color='g', ls='--', alpha=0.5, label=f'beta*_G={beta_star_G:.4f}')
    ax.set_xlabel('beta')
    ax.set_ylabel('kappa')
    ax.set_title('L=4, h=0: FIM Condition Number')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(BETA_LIST, I_eff, 'g.-')
    ax.axvline(BETA_C, color='r', ls='--', alpha=0.5)
    ax.axvline(beta_star_G, color='g', ls='--', alpha=0.5)
    ax.set_xlabel('beta')
    ax.set_ylabel('I_eff = log(kappa)')
    ax.set_title('L=4, h=0: Effective Inertia')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(BETA_LIST, Delta, 'r.-')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(BETA_C, color='r', ls='--', alpha=0.5)
    ax.axvline(beta_star_G, color='g', ls='--', alpha=0.5)
    ax.set_xlabel('beta')
    ax.set_ylabel('Delta = -dI_eff/dbeta')
    ax.set_title('L=4, h=0: Decoupling Rate')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "h0_validation_L4.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")

    # Verdict
    if abs(beta_star_G - 0.476) < 0.03:
        print("\nVERDICT: MCMC is broadly consistent with exact enumeration at this resolution.")
        print("   Treat this only as a sanity check; autocorrelation errors are not quantified.")
    # Historical failure signature from older h>0 MCMC experiments; retained only
    # as a debugging guard, not as a target value for the paper.
    elif abs(beta_star_G - 0.394) < 0.03:
        print("\nVERDICT: MCMC is far from the exact h=0 value.")
        print("   Recommendation: check thermalization, autocorrelation, and sample-size effects.")
    else:
        print(f"\nVERDICT: Unexpected value {beta_star_G:.4f}. Needs investigation.")


if __name__ == '__main__':
    main()
