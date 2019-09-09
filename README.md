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
Generate random data.
```python
X = np.random.randn(100, 3)                # 100 x 3 random std normal dist array
X = np.random.normal(1, 2, size=(100, 3))  # 100 x 3 random normal with mean 1 and stddev 2

from sklearn.datasets import make_regression, make_classification, make_blobs
# generate 100 row data for regression with 10 feature but only 5 informative
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.0, random_state=42)

# generate 100 row data for classification with 10 feature but only 5 informative with 3 classes
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=3, random_state=42)

# generate 100 row data for clustering with 10 feature with 3 cluster
X, y = make_blobs(n_samples=100, n_features=10, centers=3, cluster_std=1.0, random_state=42)
```
Load sample data
```python
from sklearn.datasets import load_boston, load_digits, load_iris
d = load_boston()
df = pd.DataFrame(d.data, columns=d.feature_names)
df['TargetCol'] = d.target
```
## Exploratory Data Analysis

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4OTYxOTI3MDQsLTE2ODU0MTA4NjQsLT
QzMzM4NDAzMiw4NTcwMzgyNTMsLTcwODIwNTU2MCwxOTI5MjIz
MzQ2LDE3ODE2OTk1MjQsODc4MTE0MzI5LC0xODQwMzM2OTcsMT
YwODg2Mzg2OSwxMzY1NjQxNTY5LDEzMDk2MzYwMTEsLTIwODkw
MTA0NzIsMTI3ODA2NDYxOF19
-->