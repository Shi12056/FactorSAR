Simulation studies
================

This directory contains all the scripts and outputs used for the simulation studies in the paper. It consists of two subdirectories:

- **`scripts/`**: All code files used in the simulation experiments.
- **`results/`**: Contains simulation data and estimation results generated during the experiments.

## Structure

### 1. Uniform Consistency of CMLEs

The following Jupyter notebooks in `scripts/` correspond to the numerical results on the uniform consistency of CMLEs in the paper:

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

### 2. Uniform Selection Consistency

The following Jupyter notebooks in `scripts/` evaluate model selection performance and correspond to the uniform selection consistency results in the paper:

```text
scripts/
├── SBM_SCAD_BIC.ipynb
├── DIM_SCAD_BIC.ipynb
└── LSM_SCAD_BIC.ipynb
```
Due to file size limits, the corresponding simulation outputs are not uploaded. However, all notebooks are reproducible — users can re-run them to regenerate the results, using the simulation settings provided in the paper.

### 3. Visualization and Summary

The following notebook is used to process simulation results and generate the figures and tables reported in the paper:

```text
scripts/
└── FSAR_Results.ipynb
```

This notebook generates:
- **Figure 1** and **Figure 2**, based on the simulation results.
- **Table 1**, summarizing the estimation and selection performance.

Relevant simulation outputs required for plotting are provided in the corresponding folders under `scripts/`.

---

**Note:** To reproduce all results, please ensure that the appropriate Python environment is set up with required packages 
(e.g., pandas，numpy，re， os，matplotlib.pyplot，time，random，ray). 

**The full list of Python packages can be found in the import section of each notebook.**


