# DESCRIBE DATA
df.info()
df.describe()
df.corr()

# GROUP BY
print(df.groupby(['State']).sum())

# TOTAL & CONT'
total = df.apply(np.sum, axis=0)
cont = df.apply(lambda x: x/sum(x), axis=0)

# PAIR PLOT
pd.plotting.scatter_matrix(data)
