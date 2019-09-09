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
# generate 1000 row data with 10 
X, y, coef = make_regression(n_samples=1000, n_features=10, n_informative=3, noise=0.0, coef=False, random_state=None)
```

#### Abc
## Exploratory Data Analysis

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4OTQ2MjM4OTYsODU3MDM4MjUzLC03MD
gyMDU1NjAsMTkyOTIyMzM0NiwxNzgxNjk5NTI0LDg3ODExNDMy
OSwtMTg0MDMzNjk3LDE2MDg4NjM4NjksMTM2NTY0MTU2OSwxMz
A5NjM2MDExLC0yMDg5MDEwNDcyLDEyNzgwNjQ2MThdfQ==
-->