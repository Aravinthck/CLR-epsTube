# Scalable and Near-Optimal $\varepsilon$-tube Clusterwise Regression (CLR) - Under review at CPAIOR 2023



## Setup 

First, install the requirements

```
pip install -r requirements.txt
```

Note that using Gurobi requires an [academic](https://www.gurobi.com/academia/academic-program-and-licenses/) or [evaluation](https://www.gurobi.com/downloads/request-an-evaluation-license/) license.



Second, setup the directory to run the codes

* Save the contents of the downloaded and unzipped folders into a single parent folder *your_folder*
* Contents include:
    * *clr_epstube* folder containing all the codes needed to run the experiments 
    * *Datasets* folder containing data files for the real datasets 
* From within *your_folder/clr_epstube*, run the config.py to make directories to save the results


<br>

## Experiments with synthetic data


All the experiments with synthetic data including plots used in the paper are in the Jupyter-notebooks organized in the [EmpricalAnalysis](EmpiricalAnalysis) folder.

* [Compare_full_MILP](EmpiricalAnalysis/Compare_full_MILP.ipynb) and [Compare_plots](EmpiricalAnalysis/Compare_plots.ipynb) for Figure 3(a)
* [Constraint_generation_example](EmpiricalAnalysis/Constraint_generation_example.ipynb) for Figure 1
* [Imbalance_expts](EmpiricalAnalysis/Imbalance_expts.ipynb) and [imb_plots](EmpiricalAnalysis/Imb_plots.ipynb) for Figure 3(b)
* [NDK_expts](EmpiricalAnalysis/NDK_expts.ipynb) and [NDK_plots](EmpiricalAnalysis/NDK_plots.ipynb) for Figure 2

<br>

## Experiments with real datasets

All the experiments for real datasets can be executed with the codes in [Real_datasets](/Real_datasets) and the saved results can be extracted and organized with the [AllResultsCollect_toLatex](Real_datasets/AllResultsCollect_toLatex.ipynb) Jupyter-notebook.

