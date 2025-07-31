Simulation Studies
================

This directory contains all the scripts and outputs used for the simulation studies in the paper. It consists of two subdirectories:

- **`scripts/`**: All code files used in the simulation experiments.
- **`results/`**: Contains simulation data and estimation results generated during the experiments.


## Structure

### 1. Uniform Consistency of CMLEs

The following Jupyter notebooks in `scripts/` correspond to the numerical results on the uniform consistency of CMLEs in the paper:

```text
scripts/
├── utils.py
├── SBM.ipynb
├── DIM.ipynb
└── LSM.ipynb

```

#### •	All supporting functions used in the experiments (such as data generation, parameter estimation, evaluation metrics, etc.) are collected in utils.py, with detailed comments provided for each function to facilitate understanding and customization.
#### •	The three notebook files correspond to different network models:
      •	SBM.ipynb: Experiments under the Stochastic Block Model (SBM)
	  •	DIM.ipynb: Experiments under the Dyad Independence Model (DIM)
	  •	LSM.ipynb: Experiments under the Latent Space Model (LSM)

Each notebook walks through the full experimental pipeline under the respective model, including simulation setup and estimation procedures. Comments are provided in each step to clarify its purpose and implementation.

Simulation results are saved under `results/`, organized into subdirectories according to the model and containing 9 combinations of sample size n and dimension p.

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
├── utils_BIC.py
├── SBM_SCAD_BIC.ipynb
├── DIM_SCAD_BIC.ipynb
└── LSM_SCAD_BIC.ipynb
```

#### •	Common utility functions used for model selection (such as SCAD penalty implementation, BIC computation, and evaluation routines) are provided in utils_BIC.py, with detailed explanations and comments for clarity.
#### •	The three notebooks conduct experiments under different network structures:
	  •	SBM_SCAD_BIC.ipynb: Model selection under the Stochastic Block Model
	  •	DIM_SCAD_BIC.ipynb: Model selection under the Dyad Independence Model
	  •	LSM_SCAD_BIC.ipynb: Model selection under the Latent Space Model

Each notebook outlines the complete workflow for model selection using penalized likelihood methods, including data generation, penalized estimation and model evaluation. Comments are provided in each step to clarify its purpose and implementation.

All notebooks are reproducible — users can re-run them to regenerate the results, using the simulation settings provided in the paper.


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

**Note:** Please ensure the required Python packages are installed before running the notebook. The full list of required packages can be found in the README in the previous level directory.

