High-Dimensional Spatial Autoregression with Latent Factors By Diversified Projections
================


This repository contains the code for replication of the results in the paper "High-Dimensional Spatial Autoregression with Latent Factors By Diversified Projections".

## Overview
The following code files can be used to reproduce simulation results presented in the main paper.

1.  A `README.md` file - This file gives a short description of the paper and an overview of how to carry out the analyses presented in our manuscript.
   
2.  A `manuscript` directory - This directory contains the source files (LaTeX) for the manuscript and any files directly related to the eneration of the manuscript, including figure files.
   
3.  A `simulation` directory - This directory contains the files for replication of the simulation results, including code, the simulated data, and the results. See `simulation/README.md` for more details.
   
4.  A `realdata` directory - This directory contains real data analysis files, including code, the real data, and the results. See `realdata/README.md` for more details.

## File Tree

Paper_FactorSAR

``
├── simulation
│   ├── KEM_SIMU.py
│   ├── Plot_Metrics_Compare.R
│   ├── __pycache__
│   ├── data_generate.py
│   ├── 【All0】simulation_study.ipynb
│   ├── 【Figure1】simulation_case_show.ipynb
│   ├── 【compare】simulation_case_show.ipynb
│   └── 【results】
├── realdata
│   ├── GMM.py
│   ├── Kmeans.py
│   └── __pycache__
└── README.md
''


