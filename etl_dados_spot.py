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

# COMMAND ----------

import pyspark.pandas as ps

# COMMAND ----------

path = 'dbfs:/FileStore/dados_spot_tratados/dados_spot_tratados.parquet/'
df_data = ps.read_parquet(path)
df_data.head()

# COMMAND ----------

df_data.describe()

# COMMAND ----------

len(df_data.year.unique())

# COMMAND ----------

df_data.year.value_counts()

# COMMAND ----------

df_data.year.value_counts().sort_index()

# COMMAND ----------

df_data.year.value_counts().sort_index().plot.bar()

# COMMAND ----------

df_data['decade'] = df_data.year.apply(lambda year: f'{(int(year)//10)*10}s')

# COMMAND ----------

df_data.head()

# COMMAND ----------

df_data_2 = df_data[['decade']]
df_data_2['qtd'] = 1

# COMMAND ----------

df_data_2 = df_data_2.groupby('decade').sum()
df_data_2

# COMMAND ----------

df_data_2.sort_index().plot.bar(y='qtd')

# COMMAND ----------

path = 'dbfs:/FileStore/dados_spot/data_by_year.csv'
df_year = ps.read_csv(path)
df_year.head()

# COMMAND ----------

len(df_year.year.unique())

# COMMAND ----------

df_year.plot.line(x='year', y='duration_ms')

# COMMAND ----------

df_year.plot.line(x='year', y=['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence'])

# COMMAND ----------

# MAGIC %md
# MAGIC **Questões à serem resolvidas baseados nos dados, para estabelecer estratégias de negócios:**
# MAGIC
# MAGIC 1 - Quem são os 10 artistas mais populares?
# MAGIC
# MAGIC 2 - Qual é o gênero musical predominante entre esses 10 artistas mais populares?
# MAGIC
# MAGIC 3 - Quais são os 10 gêneros musicais mais populares?
# MAGIC
# MAGIC 4 - Quais artistas se destacam dentro desses 10 gêneros musicais mais populares?

# COMMAND ----------

df_genres = ps.read_csv('dbfs:/FileStore/dados_spot/data_by_genres.csv')
df_artists = ps.read_csv('dbfs:/FileStore/dados_spot/data_by_artist.csv')
df_Wgenres = ps.read_csv('dbfs:/FileStore/dados_spot/data_w_genres.csv')

# COMMAND ----------

df_artists.head()

# COMMAND ----------

artista_ordenado = df_artists.sort_values(by='count', ascending=False)
artista_ordenado.head()

# COMMAND ----------

top_artista = artista_ordenado.iloc[0:10]
top_artista

# COMMAND ----------

plot_title = 'Top 10 Artistas'
top_artista.plot.bar(x='count', y = 'artists', title = plot_title)

# COMMAND ----------

lista_artistas = top_artista['artists'].unique().tolist()
lista_artistas

# COMMAND ----------

artista_genero = df_Wgenres.loc[df_Wgenres['artists'].isin(lista_artistas)]
artista_genero = artista_genero[['genres', 'artists']]
display(artista_genero)

# COMMAND ----------

df_Wgenres['qtd'] = 1  
df_Wgenres_2 = df_Wgenres[['genres', 'qtd']]  
Wgenres_2_ordenado = df_Wgenres_2.groupby('genres').sum().sort_values(by='qtd', ascending=False).reset_index() 
top_generos = Wgenres_2_ordenado.loc[0:10] 

# COMMAND ----------

plot_title = 'Top 10 Gêneros'
top_generos.plot.bar(x='qtd', y='genres', title=plot_title)

# COMMAND ----------

top_generos_2 = Wgenres_2_ordenado.loc[1:11] 

# COMMAND ----------

plot_title = 'Top 10 Gêneros'
top_generos_2.plot.bar(x='qtd', y='genres', title=plot_title)

# COMMAND ----------

lista_artistas = top_artista.artists.unique().to_list()
lista_artistas

# COMMAND ----------

artista_genero = df_Wgenres.loc[df_Wgenres['artists'].isin(lista_artistas)]
artista_genero = artista_genero[['genres', 'artists']]
display(artista_genero)
