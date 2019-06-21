# GENERATE RANDOM STD NORMAL DIST
np.random.randn(100,3)

# CREATE DATAFRAME
df = pd.DataFrame(x, columns=list('ABC'))

# READ CSV DATA
df = pd.read_csv('data.csv')

# READ SQL DATA
conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER=server_name;DATABASE=db_name;UID=username;PWD=password')
df = pd.read_sql_query('select * from table', conn)
