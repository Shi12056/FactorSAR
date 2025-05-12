Real data analysis
=========

This directory contains the data and codes used for the real data analysis in the paper.

## Data

All data files are located in the `data/` subdirectory:

- `Citydata.xlsx`: Contains 50 economic indicators for each city.
- `01matrix.xlsx`: Geographic adjacency matrix between cities.
- `Indicator group/`: Folder containing the grouping information of the economic indicators.

For detailed descriptions of the data, please refer to **Section 3.3** of the paper.


## Code

The entire analysis is implemented in the following Jupyter notebook:

- `FSAR_Realdata.ipynb`: Performs the full real data analysis, including:
  - Exploratory data analysis
  - Application of the proposed FSAR method

The notebook produces the following figures included in the paper:

- `rhocmle_se.pdf`: CMLE estimates with standard errors.
- `rhocmle_4c1.pdf`: Group-wise visualization of CMLE estimates.
- `eigen.pdf`: Plot of determing factor dimension.
- `rhofmle_com.pdf`: Comparison of CMLE and FMLE estimates.

---
**Note:** Please ensure the required Python packages are installed before running the notebook. The full list of required packages can be found in the import section of `FSAR_Realdata.ipynb`.


