# ML-Notes
Complete personal notes for performing Data Analysis, Preprocessing, and Training ML model.
## Table of contents
- [Preparation](#Preparation)
	- [Importer](#Importer)
	- [Get Data](#Get-Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)


## Preparation
### Importer
Import standard library for playing with data.
```python
import numpy as np						# numerical analysis and matrix computation 
import pandas as pd						# data manipulation and analysis on tabular data
import matplotlib.pyplot as plt			# plotting data
import seaborn as sns					# data visualization based on matplotlib
```
Import 
### Get Data
Generate random data.
```python
X = np.random.randn(100, 3)						# 100 x 3 random std normal dist array
X = np.random.normal(1, 2, size=(100, 3))		# 100 x 3 random normal with mean 1 and stddev 2

from sklearn.datasets import make_regression, make_classification, make_blobs
# generate 100 row data for regression with 10 feature but only 5 informative
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.0, random_state=42)

# generate 100 row data for classification with 10 feature but only 5 informative with 3 classes
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=3, random_state=42)

# generate 100 row data for clustering with 10 feature with 3 cluster
X, y = make_blobs(n_samples=100, n_features=10, centers=3, cluster_std=1.0, random_state=42)
```
Load sample data.
```python
from sklearn.datasets import load_boston, load_digits, load_iris
d = load_boston()										# load data dict 'like' of numpy.ndarray
df = pd.DataFrame(d.data, columns=d.feature_names)		# create dataframe with column name
df['TargetCol'] = d.target								# add TargetCol column
```
Create DataFrame from list / dict.
```python
a = [[1,2], [3,4], [5,6], [7,8]]					# from list
b = {'a': [1,2,3,4], 'b': [5,6,7,8]}				# from dictionary
c = [0,1,2,3]										# for index
df = pd.DataFrame(a, columns=list('ab'), index=c)
```
Load data from other source.
```python
df = pd.read_csv('data.csv')		# read from csv file, params: sep, index_col='col1', na_values, parse_dates)
df = pd.read_excel('data.xlsx')
```
## Exploratory Data Analysis

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NzgwODY1MzgsLTE1Nzg5MTE1OTcsLT
E2ODU0MTA4NjQsLTQzMzM4NDAzMiw4NTcwMzgyNTMsLTcwODIw
NTU2MCwxOTI5MjIzMzQ2LDE3ODE2OTk1MjQsODc4MTE0MzI5LC
0xODQwMzM2OTcsMTYwODg2Mzg2OSwxMzY1NjQxNTY5LDEzMDk2
MzYwMTEsLTIwODkwMTA0NzIsMTI3ODA2NDYxOF19
-->