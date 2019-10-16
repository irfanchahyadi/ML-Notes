# ML-Notes
Complete personal notes for performing Data Analysis, Preprocessing, and Training ML model. For easy guideline and quick copy paste snippet to real work. Fit on one page and constantly updated.
## Table of contents
- [Preparation](#Preparation)
	- [Importer](#Importer)
	- [Get Data](#Get-Data)
		- [From Other Source](#From-Other-Source)
		- [Scraping](#Scraping)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
	- [Indexing](#Indexing)
	- [Describe](#Describe)
	- [Aggregate](#Aggregate)
	- [Plotting](#Plotting)
		- Relational : [Scatter](#Scatter-plot), [Line](#Line-Plot), [Joint](#Joint-Plot), [Pair](#Pair-Plot), [Regression](#Regression-Plot)
		- Distribution : [Pie](#Pie-Plot), [Histogram](#Histogram-Plot), [Bar](#Bar-Plot), [Strip](#Strip-Plot), [Swarm](#Swarm-Plot), [Box](#Box-Plot), [Violin](#Violin-Plot), [Categorical](#Categorical-Plot)
		- Other : [Heat Map](#Heat-Map)
		- [Properties](#Properties)
- [Preprocessing](#Preprocessing)
	- [Feature Engineering](#Feature-Engineering)
	- [Missing Value](#Missing-Value)
	- [Categorical Feature](#Categorical-Feature)
	- [Transform](#Transform)
	- [Scaling and Normalize](#Scaling-and-Normalize)
- [Training Model](#Training-Model)
	- [Feature Selection](#Feature-Selection)
	- [Cross Validation](#Cross-Validation)
	- [Train Model](#Train-Model)
	- [Evaluation](#Evaluation)
	- [Hyperparameter Tuning](#Hyperparameter-Tuning)
	- [Pipeline](#Pipeline)
- [Miscellaneous](#Miscellaneous)
	- [Basic Python](#Basic-Python)
	- [Regex Cheatsheet](#Regex-Cheatsheet)
	- [Datetime Cheatsheet](#Datetime-Cheatsheet)
	- [CSS Selector Cheatsheet](#CSS-Selector-Cheatsheet)
	- [Matplotlib Cheatsheet](#Matplotlib-Cheatsheet)

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
import pyodbc                           # connect to sql server database
import pyathena                         # connect to aws athena
import gspread                          # connect to gspread
from oauth2client.service_account import ServiceAccountCredentials   # google auth
from gspread_dataframe import get_as_dataframe, set_with_dataframe   # library i/o directly from df

# Scikit-learn
from sklearn.preprocessing import Imputer, scale, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline

# Scikit-learn Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVC

# Other Tools
%reload_ext dotenv                      # reload dotenv on jupyter notebook
%dotenv                                 # load dotenv
import os                               # os interface, directory, path
import glob                             # find file on directory with wildcard
import pickle                           # save/load object on python into/from binary file
import re                               # find regex pattern on string
import scipy                            # scientific computing
import statsmodels.api as sm            # statistic lib for python
import requests                         # http for human
from bs4 import BeautifulSoup           # tool for scrape web page
```
### Get Data
Create DataFrame from list / dict.
```python
a = [[1,2], [3,4], [5,6], [7,8]]
b = {'a': [1,2,3,4], 'b': [5,6,7,8]}
c = [0,1,2,3]                                          # for index
df = pd.DataFrame(a, columns=list('ab'), index=c)      # from list
df = pd.DataFrame(b)                                   # from dictionary, default index = [0,1,2,...]
```
#### From Other Source
```python
# Read data from CSV / Excel file
df = pd.read_csv('data.csv', sep=',', index_col='col1', na_values='-', parse_dates=True)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1', usecols='A,C,E:F')

# Read data from SQL
conn  = pymysql.connect(user=user, password=pwd, database=db, host=host)       # mysql
conn  = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};
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
#### Scraping
With Requests + BeautifulSoup
```python
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'}
res = requests.get(url, headers=HEADERS)                 # request url, with user agent on headers
soup = bs4.BeautifulSoup(res.content, 'html.parser')     # create soup object
rows = soup.select('div.product')                        # selector, see misc
```
With Scrapy
```python
# Shell command:
scrapy startproject project_name           # create new project
cd project_name
scrapy genspider spider_name url           # generate new spider
scrapy crawl spider_name                   # run spider
scrapy crawl spider_name -o result.csv     # run spider, save output as csv
scrapy shell url                           # testing shell to specific url

# Spider example 1 :
class QuotesSpider(scrapy.Spider):
    name = 'quotes_1'
    start_urls = ['http://quotes.toscrape.com']

    def parse(self, response):
        self.log('I just visited: ' + response.url)

        # List of quotes
        for quote in response.css('div.quote'):
            item = {
	            'author_name': quote.css('small.author::text').extract_first(),
	            'text': quote.css('span.text::text').extract_first(),
	            'tags': quote.css('a.tag::text').extract()
	            }
            yield item

        # Follow pagination link
        rel_link = response.css('li.next > a::attr(href)').extract_first()
        if rel_link:
            abs_link = response.urljoin(rel_link)
            yield scrapy.Request(url=abs_link, callback=self.parse)
```
## Exploratory Data Analysis
### Indexing
```python
df.col1                                  # return series col1, easy way
df['col1']                               # return series col1, robust way
df[['col1', 'col2']]                     # return dataframe consist col1 and col2
df.loc[5:10, ['col1','col2']]            # return dataframe from row 5:10 column col1 and col2
df.iloc[5:10, 3:5]                       # return dataframe from row 5:10 column 3:5
df.head()                                # return first 5 rows, df.tail() return last 5 rows
df[df.col1 == 'abc']                     # filter by comparison, use ==, !=, >, <, >=, <=
df[(df.col1 == 'abc') & (df.col2 > 50)]  # conditional filter, use &(and), |(or), ~(not), ^(xor), .any(), .all()
df[df.col1.isin(['a','b'])]              # filter by is in list
df[df.col1.isnull()]                     # filter by is null, otherwise use .notnull()
df.filter(regex = 'pattern')             # filter by regex pattern, see misc
for idx, row in df.iterrows():           # iterate dataframe by rows
    print(row['col1'])                   # return index and series of row
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
df.kurt()                          # degree of peakedness, 0 normal dist, + too peaked, - almost flat
df.corr()                          # correlation matrix
df.isnull().sum()                  # count null value each column, df.isnull() = df.isna()
df.nunique()                       # unique value each column
df.sample(10)                      # return random sample 10 rows
df['col1'].value_counts(normalize=True)      # frequency each value
df.sort_values(['col1'], ascending=True)     # sort by col1 ascending, .sort_index() for index
```
### Aggregate
```python
df.sum()                           # use sum, count, median, min, mean, var, std, nunique, quantile([0.25,0.75])
df.groupby(['col1']).size()        # group by col1
df.groupby(df.col1).TargetCol.agg([np.mean, 'count'])     # multi aggregate function on group by
df.pivot(index='col1', columns='col2', values='col3')     # reshape to pivot, error when duplicate
df.pivot_table(index='col1', columns='col2', values='col3', aggfunc='sum')     # pivot table, like excel
flat = pd.DataFrame(df.to_records())            # flatten multiindex dataframe
```
### Plotting
#### Scatter plot
```python
plt.scatter(x, y, c, s)
# x, y, c, s array like object, c (color) can be color format string, s (size) can be scalar
# also df.plot.scatter(x='col1', y='col2', c='col3', s='col4') or
# sns.scatterplot(x='col1', y='col2', hue='col3', size='col4', style='col5', data=df)
```
![scatterplot](https://seaborn.pydata.org/_images/seaborn-scatterplot-13.png)
#### Line Plot
```python
plt.plot(x, y, 'ro--')
# x and y array like object, 'ro--' means red circle marker with dash line (see matplotlib cheatsheet below)
# also written as plt.plot(x, y, color='r', marker='o', linestyle='--') you can also use df.plot() or 
# sns.lineplot(x='col1', y='col2', hue='col3', size='col4', data=df)
```
![lineplot](https://matplotlib.org/_images/sphx_glr_set_and_get_001.png)
#### Joint Plot
```python
sns.jointplot(x='col1', y='col2', data=df, kind='reg')     # kind = scatter/reg/resid/kde/hex
# Joint 2 two type distrbution plot and kind plot
```
![jointplot](https://seaborn.pydata.org/_images/seaborn-jointplot-2.png)
#### Pair Plot
```python
sns.pairplot(df, x_vars=['col1'], y_vars=['col2'], hue='col3', kind='scatter', diag_kind='auto')
# multi joint plot, _vars for filter column, kind = scatter/reg, diag_kind = hist/kde
```
![pairplot](https://seaborn.pydata.org/_images/seaborn-pairplot-2.png)
#### Regression Plot
```python
regplot(x='col1', y='col2', data=df, ci=95, order=1)
# scatter plot + regression fit, ci (confidence interval 0-100), order (polynomial order)
```
![regplot](https://seaborn.pydata.org/_images/seaborn-regplot-1.png)
#### Pie Plot
```python
plt.pie(x, labels, explode, autopct='%1.1f%%')
# x, labels, explode array like also df.plot.pie(y='col1') lable get from index  
```
![pieplot](https://matplotlib.org/_images/sphx_glr_pie_features_001.png)
#### Histogram Plot
```python
plt.hist(x, bins=50, density=False,  cumulative=False)
# x array like, density (probability density), cumulative probability
# also df.plot.hist('col1') or sns.distplot(x)
```
![histplot](https://matplotlib.org/_images/sphx_glr_pyplot_text_001.png)
#### Bar Plot
```python
plt.bar(x, y)     # or plt.barh(x, y)
# x array like, also df.plot.bar(x='col1', y=['col2','col3'], stacked=True, subplots=False)
# sns.countplot(x='col1', y='col2', hue='col3', data=df, orient='v')
# sns.barplot(x='col1', y='col2', hue='col3', data=df, orient='v')
```
![barplot](https://matplotlib.org/_images/sphx_glr_barchart_001.png)
#### Strip Plot
```python
sns.stripplot(x='col1', y='col2', hue='col3', data=df, jitter=True, dodge=False, orient='v')
# for few data, jitter=True makes point not overwrite on top each other
```
![stripplot](https://seaborn.pydata.org/_images/seaborn-stripplot-4.png)
#### Swarm Plot
```python
sns.swarmplot(x='col1', y='col2', hue='col3', data=df, dodge=False, orient='v')
# for few data, more clearly than stripplot, dodge=True makes each cat in hue separable
```
![swarmplot](https://seaborn.pydata.org/_images/seaborn-swarmplot-4.png)
#### Box Plot
```python
sns.boxplot(x='col1', y='col2', hue='col3', data=df, dodge=False, orient='v')
# for large data, include median, Q1 & Q3, IQR (Q3-Q1), min (Q1-1.5*IQR), max (Q3+1.5*IQR) and outliers
```
![boxplot](https://seaborn.pydata.org/_images/seaborn-boxplot-2.png)
#### Violin Plot
```python
sns.violinplot(x='col1', y='col2', hue='col3', data=df, dodge=False, orient='v')
# kernel density plot (KDE) for visualize clearly distribution of data
```
![violinplot](https://seaborn.pydata.org/_images/seaborn-violinplot-4.png)
#### Categorical Plot
```python
sns.catplot(x='col1', y='col2', hue='col3', data=df, row='col4', col='col5', col_wrap=4, 
kind='strip', sharex=True, sharey=True, orient='v')
# categorical plot with facetgrid options 
```
![catplot](https://seaborn.pydata.org/_images/seaborn-catplot-5.png)
#### Heat Map
```python
sns.heatmap(df.corr(), annot=True, fmt='.2g', annot_kws={'size': 8}, square=True, cmap=plt.cm.Reds)
# useful for plot correlation, annot (write value data), fmt (format value), square, cmap (color map)
# other option use df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
```
![heatmap](https://seaborn.pydata.org/_images/seaborn-heatmap-1.png)
#### Properties
```python
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,4))    # subplots, access with ax[0,1]
plt.title('title')          # or ax.set_title
plt.xlabel('foo')           # or plt.ylabel, ax.set_xlabel, ax.set_ylabel
plt.xticks(x)               # x list, or ax.set_xticks
plt.xticklabels(labels)     # or ax.set_xticklabels(labels)
plt.xlim(0, 100)            # or ylim, ax.set_xlim, ax.set_ylim
plt.legend(loc='best')      # or ax.legend, loc = upper/lower/right/left/center/upper right
plt.rcParams['figure.figsize'] = (16, 10)      # setting default figsize

g = sns.FacetGrid(df, row='col1', col='col2', hue='col3')     # comparable subplot row by col1 col by col2
g.map(plt.hist, 'col4', bins=50)                              # with histogram count col4
g.map(plt.scatter, 'col4', 'col5')                            # or with scatter plot col4 and col5
```
## Preprocessing
### Feature Engineering
### Missing Value
```python
df = df.fillna(0, method=None)                                   # None/backfill/bfill/pad/ffill
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)     # strategy = mean/median/most_frequent
imp.fit(X)
X = enc.transform(X)             # perform imputing, or use .fit_transform
```
### Categorical Feature
```python
enc = LabelBinarizer()                          # label with value 0 or 1
enc = LabelEncoder()                            # label with value 0 to n-1 
enc = OrdinalEncoder()                          # label with value 0 to n-1, multi column 
enc = OneHotEncoder(handle_unknown='error')     # create dummy n column binarize, handle_unknown = error/ignore

enc.fit(X)
X = enc.transform(X)             # perform encoding, or use .fit_transform
X = enc.inverse_transform(X)     # decode back to original
```
### Transform
### Scaling and Normalize
```python
scaler = StandardScaler()                              # scale data to mean 0 and stddev 1
scaler = MinMaxScaler(feature_range=(0, 1))            # scale data to 0 to 1 (can be set as -1 to 1)
scaler = RobustScaler()quantile_range=(25.0, 75.0)     # scale data to robust to outlier
scaler = Normalizer(norm='l2')                         # normalize data

scaler.fit(X)
X = scaler.transform(X)             # perform scaling, or use .fit_transform
X = scaler.inverse_transform(X)     # scale back to original
```

## Training Model
### Feature Selection
### Cross Validation
```python
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Cross Validation
cv = cross_val_score(model, X, y, cv=5, scoring='r2', )     # (Stratified) KFold CV, return list of score
```
### Train Model
```python
# Classification model
clf = LogisticRegression(penalty='l1', C=1.0)
# C inverse regularization, smaller C stronger reg

clf = KNeighborsClassifier(n_neighbors=3)
# n_neighbors number neighbors

clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
# criterion = gini/entropy, other params: min_sample_split, min_sample_leaf, max_features
```
### Evaluation
### Hyperparameter Tuning
```python
# Grid Sarch CV: Try every parameter combination
params = {'C': [1, 0.5, 0.1, 0.05, 0.01]}
grid_cv = GridSearchCV(model, params, cv=5, scoring='r2')
grid_cv.fit(X, y)     # print grid_cv.best_params_ and grid_cv.best_score_

# Randomized Search CV: Try every parameter combination based on random distribution
params = {'C': scipy.stats.randint(0, 1)}
randomsearch_cv = RandomizedSearchCV(model, params, cv=5, scoirng='r2')
randomsearch_cv.fit(X, y)     # print grid_cv.best_params_ and grid_cv.best_score_
```
### Pipeline

## Miscellaneous
### Basic Python
#### Pickle
```python
try:                                           # check if there is pickle file
    with open('data.pickle', 'rb') as f:       # open pickle file
        data = pickle.load(f)                  # load data from pickle

except FileNotFoundError:                      # if pickle file not found
    data = some_process()                      # run some process to get data
    with open('data.pickle', 'wb') as f:       # create new blank pickle file
        pickle.dump(data, f)                   # save data to pickle
```
### Regex Cheatsheet
```python
# Cheatsheet:
\d   # any number
\D   # anything but number
\w   # any character
\W   # anything but character
.    # any character except newline
\b   # whitespace
\.   # dot character
+    # match 1 or more
*    # match 0 or 1
?    # match 0 or 1
^    # start string
$    # end string
{}   # expect 1-3, ex \d{3} expect 3 digit number, \d{1,3} expect 1-3 digit number
[]   # range, ex [A-Z] expect all capital letter, [abcd] expect letter a, b, c, d
|    # either or, ex \d{1,4}|[A-z]{4,10} expect 1-4 digit number or 4-10 character word
\n   # newline
\s   # space
\t   # tab
\e   # escape
\r   # return

# Sample:
'Chapter\s\d{1,4}[\,\.]?\d{0,1}'     # Got: Chapter 1, Chapter 23, Chapter 649, Chapter 120.5, Chapter 120,5
```
### Datetime Cheatsheet
```python
# Cheatsheet:
%Y   # Year 4 digit, ex: 1992, 2008, 2014
%y   # Year 2 digit, ex: 92, 08, 14
%m   # Month 2 digit, ex: 01, 02, ..., 12
%b   # Month abbreviation, ex: Jan, Feb, ..., Dec
%B   # Month full, ex: January, February, ..., December
%d   # Day 2 digit, ex: 01, 02, ..., 31
%w   # Weekday 1 digit, ex: 0 (Sunday), 1, ..., 6
%a   # Weekday abbreviation, ex: Sun, Mon, ..., Sat
%A   # Weekday full, ex: Sunday, Monday, ..., Saturday
%H   # Hour (24), ex: 01, 02, ..., 23, 00
%I   # Hour (12), ex: 01, 02, ..., 11, 12
%M   # Minute, ex: 00, 01, ..., 59
%S   # Second, ex: 00, 01, ..., 59
%f   # Microsecond, ex: 000000, 000001, ..., 999999
%c   # Local datetime representation, ex: Tue Aug 16 21:30:00 1988
%x   # Local date representation, ex: 08/16/88
%X   # Local time representation, ex: 21:30:00

# Sample:
'%d-%m-%Y %H:%M:%S'     # Personal preferred datetime, ex: 16-08-1988 21:30:00
```
### CSS Selector Cheatsheet
```python
# Cheatsheet:
div                # div
#abc               # id abc
.abc               # class abc
div.abc            # div with class abc
div.abc.def        # div with class both abc and def
div a              # a inside div
div > a            # a directly inside div
div + p            # p immediately after div
div ~ p            # p after div
a[target=_blank]   # a with attribute target="_blank"
```
### Matplotlib Cheatsheet
```python
# Line Styles Cheatsheet:
-    # solid line
--   # dashed line
-.   # dash-dot line
:    # dotted line

# Marker Styles Cheatsheet:
.   # point
o   # circle
^   # triangle up, also v, <, >
s   # square
p   # pentagon
*   # star
+   # plus
x   # x
|   # v line
-   # h line

# Color Styles Cheatsheet:
b   # blue
g   # green
r   # red
c   # cyan
m   # magenta
y   # yellow
k   # black
w   # white

# Cmap Cheatsheet:
viridis, plasma, Reds, cool, hot, coolwarm, hsv, Pastel1, Pastel2, Paired, Set1, Set2, Set3
plt.colormaps()     # return all possible cmap
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0Mjg4NTQwNjAsMTM0MTIwODMwNiw4OD
Q1NTA1NzcsLTIyNzg1NzQ1LC0xNTc4OTExNTk3LC0xNjg1NDEw
ODY0LC00MzMzODQwMzIsODU3MDM4MjUzLC03MDgyMDU1NjAsMT
kyOTIyMzM0NiwxNzgxNjk5NTI0LDg3ODExNDMyOSwtMTg0MDMz
Njk3LDE2MDg4NjM4NjksMTM2NTY0MTU2OSwxMzA5NjM2MDExLC
0yMDg5MDEwNDcyLDEyNzgwNjQ2MThdfQ==
-->