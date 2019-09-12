# ML-Notes
Complete personal notes for performing Data Analysis, Preprocessing, and Training ML model. For easy guideline and quick copy paste snippet to real work.
## Table of contents
- [Preparation](#Preparation)
	- [Importer](#Importer)
	- [Get Data](#Get-Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
	- [Indexing](#Indexing)
	- [Describe](#Describe)
	- [Aggregate](#Aggregate)
	- [Plotting](#Plotting)


## Preparation
### Importer
```python
# Most used
import numpy as np                      # numerical analysis and matrix computation 
import pandas as pd                     # data manipulation and analysis on tabular data
import matplotlib.pyplot as plt         # plotting data
import seaborn as sns                   # data visualization based on matplotlib

# Connection to data
import pymysql                          # connect to mysql database
import pyathena                         # connect to aws athena
import gspread                          # connect to gspread
from oauth2client.service_account import ServiceAccountCredentials   # google auth
from gspread_dataframe import get_as_dataframe, set_with_dataframe   # library i/o directly from df

# Scikit-learn
from

# Other Tools
%reload_ext dotenv                      # reload dotenv on jupyter notebook
%dotenv                                 # load dotenv
import os                               # os interface, directory, path
import glob                             # find file on directory with wildcard
import pickle                           # save/load object on python into/from binary file
import re                               # find regex pattern on string
import statsmodels.api as sm            # 
```
### Get Data
Create DataFrame from list / dict.
```python
a = [[1,2], [3,4], [5,6], [7,8]]                       # from list
b = {'a': [1,2,3,4], 'b': [5,6,7,8]}                   # from dictionary
c = [0,1,2,3]                                          # for index
df = pd.DataFrame(a, columns=list('ab'), index=c)
```
Load data from other source.
```python
# Read data from CSV / Excel file
df = pd.read_csv('data.csv', sep=',', index_col='col1', na_values='-', parse_dates=True)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1', usecols='A,C,E:F')

# Read data from SQL
conn  = pymysql.connect(user=user, password=pwd, database=db, host=host)       # mysql
conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};
       SERVER=server_name;DATABASE=db_name;UID=username;PWD=password')         # sql server
query = 'select * from employee where name = %(name)s'
df = pd.read_sql(query, conn, params={'name': 'value'})

# Read data from AWS Athena
conn  = pyathena.connect(aws_access_key_id=id, aws_secret_access_key=secret, 
                         s3_staging_dir=stgdir, region_name=region)
query = 'select * from employee'
df = pd.read_sql(query, conn)

# Read data from GSpread
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)
sheet = client.open('FileNameOnGDrive').get_worksheet(0)
df = get_as_dataframe(sheet, usecols=list(range(10)))       # use additional gspread_dataframe lib
data = sheet.get_all_values()
header = data.pop(0)
df = pd.DataFrame(data, columns=header)                     # only use gspread
```
Generate random data.
```python
X = np.random.randn(100, 3)                              # 100 x 3 random std normal dist array
X = np.random.normal(1, 2, size=(100, 3))                # 100 x 3 random normal with mean 1 and stddev 2

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
d = load_boston()                                          # load data dict 'like' of numpy.ndarray
df = pd.DataFrame(d.data, columns=d.feature_names)         # create dataframe with column name
df['TargetCol'] = d.target                                 # add TargetCol column
```
## Exploratory Data Analysis
### Indexing
```python
df.col1                                      # return series col1, easy way
df['col1']                                   # return series col1, robust way
df[['col1', 'col2']]                         # return dataframe consist only col1
df.loc[5:10, ['col1','col2']]                # return dataframe from row 5:10 column col1 and col2
df.iloc[5:10, 3:5]                           # return dataframe from row 5:10 column 3:5
df.head()                                    # return first 5 rows, df.tail() return last 5 rows
df[df.col1 == 'abc']                         # filter by comparison, use ==, !=, >, <, >=, <=
df[(df.col1 == 'abc') & (df.col2 == 'abc')]  # conditional filter, use &(and), |(or), ~(not), ^(xor), .any(), .all()
df[df.col1.isin(['a','b'])]                  # filter by is in list
df[df.col1.isnull()]                         # filter by is null, otherwise use .notnull()
df.filter(regex = 'pattern')                 # filter by regex pattern
df.

```
### Describe
```python
df.shape                           # number of rows and cols
df.columns                         # columns dataframe
df.index                           # index dataframe
df.T                               # transpose dataframe
df.info()                          # info number of rows and cols, dtype each col, memory size
df.describe(include='all')         # statistical descriptive: unique, mean, std, min, max, quartile
df.skew()                          # degree of symetrical, 0 symmetry, + righthand longer, - lefthand longer
df.kurt()                          # degree of peakedness, 0 normal dist, + flat, - peaked
df.corr()                          # correlation matrix
df.isnull().sum()                  # count null value each column, df.isnull() = df.isna()
df.nunique()                       # unique value each column
df['col1'].value_counts()          # frequency each value
df.sample(10)                      # return random sample 10 rows
df.sort_values(['col1'], ascending=True)     # sort by col1 ascending
```
### Aggregate
### Plotting

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjExNzIzNTQyMSwtMTU3ODkxMTU5NywtMT
Y4NTQxMDg2NCwtNDMzMzg0MDMyLDg1NzAzODI1MywtNzA4MjA1
NTYwLDE5MjkyMjMzNDYsMTc4MTY5OTUyNCw4NzgxMTQzMjksLT
E4NDAzMzY5NywxNjA4ODYzODY5LDEzNjU2NDE1NjksMTMwOTYz
NjAxMSwtMjA4OTAxMDQ3MiwxMjc4MDY0NjE4XX0=
-->