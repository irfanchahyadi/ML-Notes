# INDEXING
df.at[0, 'col1']				# return single value from row 0 column col1
df.iat[0, 0]					# return single value from row 0 column 0
df.loc[50:70, ['col1','col2']]	# return Series/DataFrame from row 50:70 column col1 and col2
df.iloc[50:70, 5:6]				# return Series/DataFrame from row 50:70 column 5 and 6
df['col1']                      # return series col1
df.col1                         # return series col1
df[['col1']]                    # return DataFrame consist col1

# DESCRIBE DATA
df.info()
df.describe(include='all')
df.shape
df.corr()
df.nunique()
df.isnull().sum()				# df.isnull() = df.isna()
df['col1'].value_counts()		# frequency categorical feature
df.sample(10)                   # return random sample 10 rows

# SORT BY
df.sort_values(['col1'], ascending=True)

# GROUP BY
df.groupby(['col1']).sum()
df.TargetCol.groupby(df.col1).agg([np.mean, 'count'])
size() --> series
sum(), count(), mean(), min(), median(), nunique() --> DataFrame
np.sum, np.median, np.min, np.max, np.std

# PIVOT
grouped = df.groupby(['col1', 'TargetCol']).size().reset_index()
pivoted = grouped.pivot(index='col1', columns='TargetCol')

# FLATTEN MULTI INDEX
flat = pd.DataFrame(df.to_records())

# TOTAL & CONT'
total = df.apply(np.sum, axis=0)
cont = df.apply(lambda x: x/sum(x), axis=0)

# PLOTTING
plt.figure(figsize=(12,10))
plot = df.plot()
hist = df.hist(bins=50, figsize=(16, 10))
stripplot = sns.stripplot(x='col1', y='col2', data=df, jitter=True)
swarmplot = sns.swarmplot(x='col1', y='col2', data=df)
boxplot = df.boxplot()
boxplot = df.boxplot(by='col1')			# group by categorical column col1
boxplot = sns.boxplot(x='col1', y='TargetCol', data=df)
violinplot = sns.violinplot(x='col1', y='col2', data=df)
jointplot = sns.jointplot(x='col1', y='col2', data=df, kind='kde')
pairplot = pd.plotting.scatter_matrix(data)
pairplot = sns.pairplot(df, x_vars=['col1'], y_vars='TargetCol')
distplot = sns.distplot(df.TargetCol, bins=100)
countplot = df['col1'].value_counts().plot(kind='bar')
countplot = sns.countplot(x='col1', data=df)
stackedbar = pivoted.plot.bar(stacked=True)

# FACETGRID
g = sns.FacetGrid(df, col='TargetCol')
g.map(sns.distplot, 'col1')
g.map(sns.countplot, 'catcol')
g.map(sns.barplot, 'catcol', 'numcol')

# CORRELATION MATRIX PLOT
corrplot = sns.heatmap(df.corr(), cmap=plt.cm.Reds, annot=True)
corrplot = sns.heatmap(corr[abs(corr) > 0.5], annot=True, annot_kws={'size': 8}, square=True, cmap=plt.cm.viridis)
corrplot = plt.matshow(df.corr())
df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)

# PAIR PLOT TO ALL NUMERIC FEATURES
df_num = df.select_dtypes(['int64', 'float64'])
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(df_num, x_vars=df_num.columns[i:i+5], y_vars='TargetCol')

# REGPLOT ALL NUMERIC FEATURES
fig, ax = plt.subplots(int(np.ceil(len(features_list)/3)),3, figsize=(16, 10))
for i, ax in enumerate(fig.axes[:-1]):
    sns.regplot(x=features_list[i], y='TargetCol', data=df, ax=ax)

# COUNTPLOT ALL CATEGORICAL FEATURES
fig, ax = plt.subplots(int(np.ceil(len(categ_features)/3)), 3, figsize=(16, 30))
for i, ax in enumerate(fig.axes[:-1]):
    sns.countplot(x=categ_features[i], data=df, ax=ax)
