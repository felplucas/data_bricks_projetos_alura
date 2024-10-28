# Databricks notebook source
dbutils.fs.ls("dbfs:/FileStore/dados_spot")

# COMMAND ----------

caminho_data = "dbfs:/FileStore/dados_spot/data.csv"

# COMMAND ----------

df_data = spark.read.csv(caminho_data, inferSchema=True, header=True)

# COMMAND ----------

df_data.show(5)

# COMMAND ----------

# Ensure df_data is a Spark DataFrame
df_data = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/dados_spot/data.csv")

# Convert Spark DataFrame to Pandas API on Spark DataFrame
df_data_pandas = df_data.pandas_api()

# Display the DataFrame
display(df_data_pandas)

# COMMAND ----------

type(df_data)

# COMMAND ----------

df_data_pandas.head()

# COMMAND ----------

df_data_pandas.info()

# COMMAND ----------

colunas_float = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

colunas_int = ['duration_ms', 'mode', 'popularity', 'key', 'explicit']

# COMMAND ----------

df_data_pandas[colunas_float] = df_data_pandas[colunas_float].astype(float)
df_data_pandas[colunas_int] = df_data_pandas[colunas_int].astype(int)

# COMMAND ----------

df_data_pandas.info()

# COMMAND ----------

df_data_pandas.head()

# COMMAND ----------

type(df_data_pandas.artists.iloc[0])

# COMMAND ----------

# Select the first 9 rows
X = df_data.limit(9)

# Display the result
display(X)

# COMMAND ----------

from pyspark.sql.functions import regexp_replace

X = X.withColumn(
    "artists",
    regexp_replace("artists", r"\[|\]|\'", "")
)

display(X)

# COMMAND ----------

from pyspark.sql.functions import regexp_replace

X = X.withColumn("artists", regexp_replace("artists", r"\[|\]|\'", ""))
display(X)

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col

X = X.withColumn('artists', regexp_replace(col('artists'), r"\[|\]|\'", ""))
X = X.withColumn('artists', regexp_replace(col('artists'), ",", ";"))

display(X)

# COMMAND ----------

df_data_pandas['artists'] = df_data_pandas.artists.str.replace("\[|\]|\'", "")
df_data_pandas['artists'] = df_data_pandas.artists.str.replace(",", ";")

# COMMAND ----------

dbutils.fs.ls('/FileStore')

# COMMAND ----------

dbutils.fs.mkdirs('/FileStore/dados_spot_tratados')

# COMMAND ----------

dbutils.fs.ls('/FileStore')

# COMMAND ----------

df_data_pandas.to_parquet('/FileStore/dados_spot_tratados/dados_spot_tratados.parquet')

# COMMAND ----------

dbutils.fs.ls('/FileStore/dados_spot_tratados')