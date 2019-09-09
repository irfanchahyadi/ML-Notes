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
X, y, coef = make_regression(n_samples=1000, n_features=10, n_informative=3, noise=0.0, coef=False, random_state=None)
```

#### Abc
## Exploratory Data Analysis

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY0OTYzNzYyNiw4NTcwMzgyNTMsLTcwOD
IwNTU2MCwxOTI5MjIzMzQ2LDE3ODE2OTk1MjQsODc4MTE0MzI5
LC0xODQwMzM2OTcsMTYwODg2Mzg2OSwxMzY1NjQxNTY5LDEzMD
k2MzYwMTEsLTIwODkwMTA0NzIsMTI3ODA2NDYxOF19
-->