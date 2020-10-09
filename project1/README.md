# Project 1
Contains all files used in project 1:

[regression.py](regression.py): Contains implementations of OLS and Ridge regression,
as well as design matrix algorithm. Also includes Lasso regression by way of sklearn.

[bias_variance.py](bias_variance.py): Computes bias-variance
analysis for regression models over polynomial degrees, using bootstrap resampling

[k_folds.py](k_folds.py): Computes the mean squared error of a model for
different polynomial degrees, using k-fold cross-validation

[grid_search.py](grid_search.py): Contains convenience methods for running
1D and 2D searches for shrinkage parameters and/or polynomial degrees.

task_a - task_g: Contains code solving each task, with run examples when relevant,
uses mostly bias_variance.py, k_folds.py and grid_search.py.

[tools.py](tools.py): Contains various implementations of dataset processing tools,
now only scaler is really used (sklearn is used for train/test split, etc.)

[statistics.py](statistics.py): Contains convenience methods for computing MSE
and R2-score.

[terrain.py](terrain.py): Loads downsampled terrain dataset, used in task_g.
Also creates figures of terrain when run as main.

[franke_function.py](franke_function.py): Creates franke function dataset.
Also creates figures of the Franke function when run as main.

[test_regression.py](test_regression.py): Contains crude tests, plan to expand
to full unit tests using pytest in future projects.
