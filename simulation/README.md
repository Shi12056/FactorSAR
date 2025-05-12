Simulation studies
================

This directory contains all the scripts and outputs used for the simulation studies in the paper. It consists of two subdirectories:

- **`scripts/`**: All code files used in the simulation experiments.
- **`results/`**: Contains simulation data and estimation results generated during the experiments.

## Structure

### 1. Consistency of CMLE Results

The following Jupyter notebooks in `scripts/` correspond to the numerical results on consistency of CMLE in the paper:

```text
scripts/
├── SBM.ipynb
├── DIM.ipynb
└── LSM.ipynb

```
Each notebook simulates data under a specific network model and computes the estimation results. The corresponding simulation results are saved in subdirectories under `results/`, where each subdirectory contains 9 different combinations of n and p.

```text
results/
├── Results_Block_02/
│   ├── n500_p50_B500/
│   ├── n500_p100_B500/
│   └── ...
├── Results_dep_02/
│   ├── n1000_p50_B500/
│   └── ...
└── Results_LSM_02/
    ├── n1500_p200_B500/
    └── ...
```
