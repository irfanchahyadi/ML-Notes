# COPY DATAFRAME
dx = df[['col1', 'col2']].copy()

# DROP COLUMN
df.drop('col1', axis=1, inplace=True)

# CREATE COLUMN
df['newcol'] = 1

# CREATE STD DEVIATION EACH COLUMN
df.apply(lambda x: abs(x-x.mean())/x.std(), axis=0)
