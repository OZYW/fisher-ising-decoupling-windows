# Parameterization-Dependent Decoupling Windows in Statistical Manifolds

This repository is the official reproducibility package for:

> **Parameterization-Dependent Decoupling Windows in Statistical Manifolds: An Empirical Study via Finite Ising Models**

All core claims in the manuscript can be reproduced from the deterministic exact-enumeration pipeline included here.  The central result is that Fisher-information condition-number windows are coordinate-dependent diagnostics, not Riemannian invariants.

## Artifact Scope

- Exact enumeration over all `2^16 = 65536` spin states for the main finite Ising-model evidence.
- Reproduction scripts for all three manuscript figures.
- CSV, NPZ, and pickle outputs corresponding to the exact-enumeration tables and plotted curves.
- Optional MCMC drivers retained for transparency; these are not required for the manuscript's core exact-enumeration claims.

## Main Results

- Exact enumeration at `N=16` and `h=0.01` gives `beta*_G = 0.4789 +/- 0.0258` in `P1=(beta,h)` across five non-complete topologies.
- In `P2=(beta,B=beta*h)`, only three of five non-complete topologies have a crossing, with `beta*_G = 0.2431 +/- 0.0113` over those valid crossings.
- A zero-field exact check gives the same crossing pattern and nearly identical values, so the weak field is not driving the parameterization-dependence result.
- The scalar curvature calculation agrees across coordinates to median relative error `4.1e-4` and is smooth for the finite `Grid 4x4` system.

## Repository Structure

```text
.
├── manuscript.tex
├── CITATION.cff
├── requirements.txt
├── requirements-lock.txt
├── requirements-optional.txt
├── src/
│   ├── run_exact_decoupling_windows.py          # Figure 1 and main exact tables
│   ├── run_exact_scalar_curvature_grid4x4.py    # Figure 2 scalar-curvature benchmark
│   ├── plot_mcmc_noise_diagnostic.py            # Figure 3 from recorded pilot data
│   ├── run_mcmc_fss_optional.py                 # Optional long MCMC FSS driver
│   ├── run_mcmc_scalar_curvature_optional.py    # Optional long R(beta) MCMC driver
│   ├── check_zero_field_validation_optional.py   # Optional h=0 MCMC validation driver
│   └── legacy_control_experiment_n16.py         # Legacy control experiment; not used in final manuscript
├── figures/
│   ├── figure1_exact_decoupling_windows.png
│   ├── figure2_scalar_curvature_grid4x4.png
│   └── figure3_mcmc_noise_diagnostic.png
└── data/
    ├── mcmc_noise_L4_pilot.csv
    ├── exact_decoupling_summary.csv
    ├── exact_decoupling_rescan.npz
    ├── exact_decoupling_rescan.pkl
    └── .cache_n16_curvature/results.pkl
```

## Requirements

Python 3.9 or newer is recommended.

```bash
python3 -m pip install -r requirements.txt
```

The exact environment used for the final local verification is recorded in `requirements-lock.txt`.  The optional MCMC scripts additionally require PyTorch and are not needed to reproduce the exact-enumeration claims.

```bash
python3 -m pip install -r requirements-optional.txt
```

## Reproduce the Evidence

Run commands from the repository root.

### Figure 1 / Main Tables: exact decoupling-window scan

```bash
python3 src/run_exact_decoupling_windows.py
```

Outputs:

- `figures/figure1_exact_decoupling_windows.png`
- `data/exact_decoupling_summary.csv`
- `data/exact_decoupling_rescan.npz`
- `data/exact_decoupling_rescan.pkl`

This script is the deterministic source for the manuscript's `beta*_G` tables and the `h=0` robustness check.

### Figure 2: scalar-curvature benchmark

```bash
python3 src/run_exact_scalar_curvature_grid4x4.py
```

Outputs:

- `figures/figure2_scalar_curvature_grid4x4.png`
- `data/.cache_n16_curvature/results.pkl`

This computes `R(beta)` independently in `P1=(beta,h)` and `P2=(beta,B)` on the periodic `Grid 4x4`.

### Figure 3: recorded MCMC noise diagnostic

```bash
python3 src/plot_mcmc_noise_diagnostic.py
```

Outputs:

- `figures/figure3_mcmc_noise_diagnostic.png`

This script reads the recorded pilot data in `data/mcmc_noise_L4_pilot.csv`.  It does not rerun MCMC.  The long MCMC driver that produced this type of data is `src/run_mcmc_scalar_curvature_optional.py`.

## Notes on Reproducibility

- The main exact-enumeration scripts are deterministic and enumerate all `2^16 = 65536` spin states.
- The `Random Regular` topology is generated as a true simple 4-regular graph with a fixed seed.
- The `Grid 4x4` topology uses periodic boundary conditions.
- The manuscript treats `h=0.01` as a weak-field finite-size diagnostic path, not as the zero-field thermodynamic Ising point.
- The zero-field exact check is included in `src/run_exact_decoupling_windows.py`.

## Citation

GitHub reads `CITATION.cff` and exposes a repository citation through the "Cite this repository" button.  A BibTeX citation is also provided here for convenience:

```bibtex
@article{ye2026decoupling,
  title={Parameterization-Dependent Decoupling Windows in Statistical Manifolds:
         An Empirical Study via Finite Ising Models},
  author={Ye, Wei},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/OZYW/fisher-ising-decoupling-windows}
}
```
