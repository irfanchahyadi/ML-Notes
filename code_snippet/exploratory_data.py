# INDEXING
df.at[0, 'col1']				# return single value from row 0 column col1
df.iat[0, 0]					# return single value from row 0 column 0
df.loc[50:70, ['col1','col2']]	# return Series/DataFrame from row 50:70 column col1 and col2
df.iloc[50:70, 5:6]				# return Series/DataFrame from row 50:70 column 5 and 6

# DESCRIBE DATA
df.info()
df.describe()
df.shape
df.corr()
df.nunique()
df.isnull().sum()

# SORT BY
df.sort_values(['col1'], ascending=True)

# GROUP BY
df.groupby(['col1']).sum()

# TOTAL & CONT'
total = df.apply(np.sum, axis=0)
cont = df.apply(lambda x: x/sum(x), axis=0)

# PLOTTING
plt.rcParams['figure.figsize'] = (16, 10)
plt.figure(figsize=(12,10))
hist = df.hist(bins=50, figsize=(16, 20))
plot = df.plot()
boxplot = df.boxplot()
pairplot = pd.plotting.scatter_matrix(data)
distplot = sns.distplot(df.TargetCol, bins=100)

# CORRELATION MATRIX PLOT
cor = sns.heatmap(df.corr(), cmap=plt.cm.Reds, annot=True)
corr_plot = plt.matshow(data.corr())
data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)

# PAIR PLOT TO ALL FEATURE
df_num = df.select_dtypes(['int64', 'float64'])
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(df_num, x_vars=df_num.columns[i:i+5], y_vars='TargetCol')