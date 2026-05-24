"""Generate the MCMC scalar-curvature noise-diagnosis plot.

This script does not rerun MCMC.  It reads the recorded L=4 pilot values from
../data/mcmc_noise_L4_pilot.csv and recreates the diagnostic figure used
in the paper.  The underlying MCMC driver is run_mcmc_scalar_curvature_optional.py.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BETA_C = np.log(1.0 + np.sqrt(2.0)) / 2.0  # Onsager exact ~0.4407
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mcmc_noise_L4_pilot.csv"


def load_pilot_data(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["beta"], data["R"]


def make_diagnosis_plot(out_path: Path, beta: np.ndarray, r_values: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- Left panel: R(beta) scatter showing sign flips ----------------------
    ax = axes[0]
    pos_mask = r_values > 0
    neg_mask = r_values < 0

    ax.scatter(beta[pos_mask], r_values[pos_mask], s=70, color='C0',
               edgecolor='k', linewidth=0.6, label=r'$R > 0$', zorder=3)
    ax.scatter(beta[neg_mask], r_values[neg_mask], s=70, color='C3',
               edgecolor='k', linewidth=0.6, label=r'$R < 0$', zorder=3)
    ax.plot(beta, r_values, '-', color='gray', alpha=0.4, linewidth=1, zorder=1)

    ax.axhline(0, color='k', linewidth=0.6, alpha=0.5)
    ax.axvline(BETA_C, color='green', linestyle='--', linewidth=1,
               alpha=0.7, label=fr'$\beta_c = {BETA_C:.4f}$')
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel(r'$R(\beta)$  [MCMC, $L{=}4$]', fontsize=12)
    ax.set_title(r'(a) MCMC $R(\beta)$ at $L=4$: noise-dominated, sign-flipping',
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate the absurd magnitudes
    ax.annotate(r'$|R|\sim 10^5$', xy=(0.4914, -127246), xytext=(0.46, -90000),
                fontsize=10, color='C3',
                arrowprops=dict(arrowstyle='->', color='C3', alpha=0.7))
    ax.annotate('peak at $\\beta_c$\nbut isolated', xy=(0.4400, 82522),
                xytext=(0.395, 60000), fontsize=10, color='C0',
                arrowprops=dict(arrowstyle='->', color='C0', alpha=0.7))

    # --- Right panel: SNR analysis -------------------------------------------
    ax = axes[1]
    delta_range = np.linspace(0.0005, 0.01, 200)

    # SNR = N_eff * delta^4; plot N_eff required for SNR=1
    N_required = 1.0 / (delta_range ** 4)

    ax.loglog(delta_range, N_required, color='C2', linewidth=2,
              label=r'$N_{\mathrm{eff}}$ for $\sigma_R/R = 1$')
    ax.axvline(0.002, color='C3', linestyle='--', alpha=0.7,
               label=r'used $\delta = 0.002$')
    ax.axhline(5000, color='C0', linestyle=':', alpha=0.7,
               label=r'achieved $N_{\mathrm{samples}} = 5{,}000$')

    # Highlight the gap
    ax.fill_between(delta_range, N_required,
                    np.full_like(N_required, 5000),
                    where=(N_required > 5000),
                    alpha=0.15, color='C3',
                    label='infeasible region')

    ax.set_xlabel(r'stencil step $\delta$', fontsize=12)
    ax.set_ylabel(r'$N_{\mathrm{eff}}$ required', fontsize=12)
    ax.set_title(r'(b) SNR floor: $N_{\mathrm{eff}} > 1/\delta^4$ for $R$-recovery',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # Annotate the gap at delta=0.002:
    #   required N_eff at delta=0.002 is 1/delta^4 = 1/(2e-3)^4 = 6.25e10
    #   achieved N_samples = 5000
    #   ratio = 6.25e10 / 5e3 = 1.25e7  -> log10 ~ 7.1
    ax.annotate('', xy=(0.002, 6.25e10), xytext=(0.002, 5000),
                arrowprops=dict(arrowstyle='<->', color='C3', alpha=0.85, lw=2))
    ax.text(0.0023, 1.8e7, r'gap $\sim 10^{7}$',
            fontsize=11, color='C3',
            bbox=dict(facecolor='white', edgecolor='C3', alpha=0.85, pad=2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    BETA, R = load_pilot_data()
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'figure3_mcmc_noise_diagnostic.png'
    make_diagnosis_plot(out_path, BETA, R)

    # Print summary stats
    print("\n=== L=4 R(beta) Summary ===")
    print(f"  N points:     {len(R)}")
    print(f"  R range:      [{R.min():+.0f}, {R.max():+.0f}]")
    print(f"  R median:     {np.median(R):+.0f}")
    print(f"  R |median|:   {np.median(np.abs(R)):+.0f}")
    print(f"  Sign flips:   {np.sum(np.diff(np.sign(R)) != 0)}/14")
    print(f"  Pos/Neg:      {np.sum(R > 0)}/{np.sum(R < 0)}")
    print(f"\n  Predicted noise floor: sigma_R/R ~ {0.014/(0.002)**2:.0f}")
    print(f"  Required N_eff for SNR=1 at delta=0.002: {1/(0.002**4):.2e}")
