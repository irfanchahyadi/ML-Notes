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
df.drop(['col1'], axis=1, inplace=True)

# DROP ROW CONSIST NA
df.dropna()

# CREATE COLUMN
df['newcol'] = 1

# REPLACE VALUE
df['Sex'] = df.Sex.map({'male': 1, 'female': 0})

# APPLY AND APPLYMAP
def num_only(x):
    if type(x) in (int, float):
        return x
    else:
        return 0

df.apply(lambda x: x*2)		# apply function along axis (default axis=0 row)
df.applymap(num_only)		# apply function element wise

# CREATE STD DEVIATION EACH COLUMN
df.apply(lambda x: abs(x-x.mean())/x.std(), axis=0)

# GET FIRST LETTER ON FEATURE
df[['col1']].applymap(lambda x: 'Z' if pd.isnull(x) else x[0])

# SELECT FEATURE CORRELATE TO TARGET FEATURE
corr = df.corr()
corr_tgt = abs(corr['TargetCol'])
relevant_features = corr_tgt[corr_tgt > 0.5]

# CLEAN ZERO VALUE AND SELECT HIGH CORRELATE FEATURE
individual_features = []
for i in df.columns:
    temp_df = df[[i, 'TargetCol']]
    temp_df = temp_df[temp_df[i] != 0]
    individual_features.append(temp_df)

corr_dict = {}
for feature in individual_features:
    corr_temp = feature.corr().iat[0,1]
    if corr_temp > 0.5:
        corr_dict[feature.columns[0]] = corr_temp
relevant_features3 = pd.Series(corr_dict)

# IMPUTE WITH GROUP BY
def impute_median(series):
    return series.fillna(series.median())
titanic.age = titanic.groupby(['sex', 'pclass'])['age'].transform(impute_median)

# GET DUMMIES
df = pd.get_dummies(df, columns=['col1'], prefix='col')
