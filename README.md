# ML-Notes
Complete notes for performing Data Gathering, EDA, Preprocessing, Training ML model, evaluating model and hyperparameter tuning.
## Table of contents
- [Preparation](#Preparation)
	- [Importer](#Importer)
	- [Get Data](#Get-Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)


## Preparation
### Importer
Import standard library for playing with data.
```python
import numpy as np                    # numerical analysis and matrix computation 
import pandas as pd                   # data manipulation and analysis on tabular data
import matplotlib.pyplot as plt       # plotting data
import seaborn as sns                 # data visualization based on matplotlib
```
Import 
### Get Data
Generate random data
```python
X = np.random.randn(100, 3)                # 100 x 3 random std normal dist array
X = np.random.normal(1, 2, size=(100, 3))  # 100 x 3 random normal with mean 1 and stddev 2

from sklearn.datasets import make_regression, make_classification, make_blobs
# generate 100 row data for regression with 10 feature but only 5 informative with 0.0 noise
X, y, coef = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.0, coef=True, random_state=42)
# generate 100 row data for classification with 10 feature but only 5 informative with 0.0 noise
X, y, coef = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
```

#### Abc
## Exploratory Data Analysis

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2ODU0MTA4NjQsLTQzMzM4NDAzMiw4NT
cwMzgyNTMsLTcwODIwNTU2MCwxOTI5MjIzMzQ2LDE3ODE2OTk1
MjQsODc4MTE0MzI5LC0xODQwMzM2OTcsMTYwODg2Mzg2OSwxMz
Y1NjQxNTY5LDEzMDk2MzYwMTEsLTIwODkwMTA0NzIsMTI3ODA2
NDYxOF19
-->