# LIST COLUMN
cols = df.keys().to_series().groupby(df.dtypes).groups
cols = {k.name :v for k, v in cols.items()}
# or just select based on dtypes
df.select_dtypes(['object'])
# list dtypes
dtypes = [i.name for i in set(df.dtypes.tolist())]

# COPY DATAFRAME
dx = df[['col1', 'col2']].copy()

# DROP COLUMN
df.drop('col1', axis=1, inplace=True)

# CREATE COLUMN
df['newcol'] = 1

# CREATE STD DEVIATION EACH COLUMN
df.apply(lambda x: abs(x-x.mean())/x.std(), axis=0)

# SELECT FEATURE CORRELATE TO TARGET FEATURE
corr = df.corr()
corr_tgt = abs(corr['TargetCol'])
relevant_features = corr_tgt[corr_tgt > 0.5]