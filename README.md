High-Dimensional Spatial Autoregression with Latent Factors by Diversified Projections
================


This repository contains the code for replication of the results in the paper "High-Dimensional Spatial Autoregression with Latent Factors by Diversified Projections".

## Overview
The following code files can be used to reproduce simulation results and the real data analysis presented in the main paper.

1.  A `README.md` file - This file gives a short description of the paper and an overview of how to carry out the analyses presented in our manuscript.
   
2.  A `manuscript` directory - This directory contains the source files (LaTeX) for the manuscript and any files directly related to the eneration of the manuscript, including figure files.
   
3.  A `simulation` directory - This directory contains the files for replication of the simulation results, including code, the simulated data, and the results. See `simulation/README.md` for more details.
   
4.  A `realdata` directory - This directory contains real data analysis files, including code, the real data, and the results. See `realdata/README.md` for more details.

## Requirements

Before running the scripts, you need python==3.12.4 and the following python packages:
- numpy=2.0.0
- pandas=2.2.2
- scipy=1.14.0
- statsmodels=0.14.5
- seaborn=0.13.2
- scikit-learn=1.5.0
- ray=2.48.0
- matplotlib=3.9.0

For your ease, my conda environment is exported as file. You can easily restore the environment by typing command: 

    conda env create -f environment.yaml

It would take you about ten minutes to download all necessary python packages.
