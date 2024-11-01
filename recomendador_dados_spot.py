# Databricks notebook source
import pyspark.pandas as ps

# COMMAND ----------

path = '/FileStore/dados_spot_tratados/dados_spot_tratados.parquet'
df_data = spark.read.parquet(path)
display(df_data)

# COMMAND ----------

df_data.printSchema()

# COMMAND ----------

df_data = df_data.dropna()

# COMMAND ----------

df_data.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F

df_data = df_data.withColumn('artists_song', F.concat(df_data.artists, F.lit(' - '), df_data.name))

# COMMAND ----------

display(df_data)

# COMMAND ----------

X = df_data.columns
X.remove('artists')
X.remove('id')
X.remove('name')
X.remove('artists_song')
X.remove('release_date')
X

# COMMAND ----------

# Display the Spark DataFrame
display(df_data)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# Convert the 'year' column to integer
df_data = df_data.withColumn("year", df_data["year"].cast("integer"))

# Apply VectorAssembler
dados_encoded_vector = VectorAssembler(
    inputCols=X, 
    outputCol='features'
).transform(df_data)

display(dados_encoded_vector)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# COMMAND ----------

scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
model_scaler = scaler.fit(dados_encoded_vector)
dados_musicas_scaler = model_scaler.transform(dados_encoded_vector)

# COMMAND ----------

dados_musicas_scaler.select('features_scaled').show(truncate=False, n=5)

# COMMAND ----------

k = len(X)
k

# COMMAND ----------

from pyspark.ml.feature import PCA

# COMMAND ----------

pca = PCA(k=k, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dados_musicas_scaler)
dados_musicas_pca = model_pca.transform(dados_musicas_scaler)

# COMMAND ----------

sum(model_pca.explainedVariance) * 100

# COMMAND ----------

lista_valores = [sum(model_pca.explainedVariance[0:i+1]) for i in range(k)]
lista_valores

# COMMAND ----------

import numpy as np

# COMMAND ----------

k = sum(np.array(lista_valores) <= 0.7)
k

# COMMAND ----------

pca = PCA(k=6, inputCol='features_scaled', outputCol='pca_features')
model_pca = pca.fit(dados_musicas_scaler)
dados_musicas_pca_final = model_pca.transform(dados_musicas_scaler)

# COMMAND ----------

dados_musicas_pca_final.select('pca_features').show(truncate=False, n=5)

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

pca_pipeline = Pipeline(stages=[VectorAssembler(inputCols=X, outputCol='features'),
StandardScaler(inputCol='features', outputCol='features_scaled'),
PCA(k=6, inputCol='features_scaled', outputCol='pca_features')])

# COMMAND ----------

model_pca_pipeline = pca_pipeline.fit(df_data)

# COMMAND ----------

projection = model_pca_pipeline.transform(df_data)

# COMMAND ----------

projection.select('pca_features').show(truncate=False, n=5)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

SEED = 1224

# COMMAND ----------

kmeans = KMeans(k=50, featuresCol='pca_features', predictionCol='cluster_pca', seed=SEED)

# COMMAND ----------

modelo_kmeans = kmeans.fit(projection)

# COMMAND ----------

projetion_kmeans = modelo_kmeans.transform(projection) 

# COMMAND ----------

projetion_kmeans.select(['pca_features','cluster_pca']).show()

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

# COMMAND ----------

projetion_kmeans = projetion_kmeans.withColumn('x', vector_to_array('pca_features')[0])\
                                   .withColumn('y', vector_to_array('pca_features')[1])

# COMMAND ----------

projetion_kmeans.select(['x', 'y', 'cluster_pca', 'artists_song']).show()

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

fig = px.scatter(projetion_kmeans.toPandas(), x='x', y='y', color='cluster_pca', hover_data=['artists_song'])
fig.show()

# COMMAND ----------

nome_musica = 'Taylor Swift - Blank Space'

# COMMAND ----------

cluster = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
cluster

# COMMAND ----------

cluster = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
cluster

# COMMAND ----------

musicas_recomendadas = projetion_kmeans.filter(projetion_kmeans.cluster_pca == cluster)\
.select('artists_song', 'id', 'pca_features')
musicas_recomendadas.show()

# COMMAND ----------

componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
.select('pca_features').collect()[0][0]
componenetes_musica                             

# COMMAND ----------

from scipy.spatial.distance import euclidean
from pyspark.sql.types import FloatType
import pyspark.sql.functions as f

# COMMAND ----------

def calcula_distance(value):
    return euclidean(componenetes_musica, value)

udf_calcula_distance = f.udf(calcula_distance, FloatType())

musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distance('pca_features'))

# COMMAND ----------

recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

recomendadas.show()

# COMMAND ----------

def recomendador(nome_musica):
    cluster = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
    musicas_recomendadas = projetion_kmeans.filter(projetion_kmeans.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
    componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]

    def calcula_distance(value):
        return euclidean(componenetes_musica, value)

    udf_calcula_distance = f.udf(calcula_distance, FloatType())

    musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distance('pca_features'))

    recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

    return recomendadas

# COMMAND ----------

df_recomedada = recomendador('Taylor Swift - Blank Space')
df_recomedada.show()

# COMMAND ----------

!pip install spotipy

# COMMAND ----------

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

# COMMAND ----------

scope = "user-library-read playlist-modify-private"

OAuth = SpotifyOAuth(
        scope=scope,         
        redirect_uri='http://localhost:5000/callback',
        client_id = 'fa299e478df943e28d25d45a20fee79c',
        client_secret = '2376ff31e51a44c8adf3eff5d56fed65') 

# COMMAND ----------

client_credentials_manager = SpotifyClientCredentials(client_id = 'fa299e478df943e28d25d45a20fee79c',
                                                      client_secret = '2376ff31e51a44c8adf3eff5d56fed65')

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# COMMAND ----------

id = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('id').collect()[0][0]
id

# COMMAND ----------

sp.track(id)

# COMMAND ----------

playlist_id = df_recomedada.select('id').collect()

# COMMAND ----------

playlist_track = []
for id in playlist_id:
    playlist_track.append(sp.track(id[0]))

# COMMAND ----------

def recomendador(nome_musica):
    cluster = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
    musicas_recomendadas = projetion_kmeans.filter(projetion_kmeans.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
    componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]

    def calcula_distance(value):
        return euclidean(componenetes_musica, value)

    udf_calcula_distance = f.udf(calcula_distance, FloatType())

    musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distance('pca_features'))

    recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

    id = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('id').collect()[0][0]

    playlist_id = recomendadas.select('id').collect()

    playlist_track = []

    for id in playlist_id:
        playlist_track.append(sp.track(id[0]))

    return len(playlist_track)

# COMMAND ----------

recomendador('Taylor Swift - Blank Space')

# COMMAND ----------

!pip install scikit-image

# COMMAND ----------

import matplotlib.pyplot as plt
from skimage import io

nome_musica = 'Taylor Swift - Blank Space'

id = projetion_kmeans\
          .filter(projetion_kmeans.artists_song == nome_musica)\
          .select('id').collect()[0][0]

track = sp.track(id)

url = track["album"]["images"][1]["url"]
name = track["name"]

image = io.imread(url)
plt.imshow(image)
plt.xlabel(name, fontsize = 10)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
from skimage import io

def visualize_songs(name,url):

    plt.figure(figsize=(15,10))
    columns = 5
    for i, u in enumerate(url):
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)
        image = io.imread(u)
        plt.imshow(image)
        ax.get_yaxis().set_visible(False)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(name[i], fontsize = 10)
        plt.tight_layout(h_pad=0.7, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.grid(visible=None)
    plt.show()

# COMMAND ----------

playlist_id = df_recomedada.select('id').collect()

name = []
url = []
for i in playlist_id:
    track = sp.track(i[0])
    url.append(track["album"]["images"][1]["url"])
    name.append(track["name"])

# COMMAND ----------

visualize_songs(name,url)

# COMMAND ----------

def recomendador(nome_musica):
  # Calcula musicas recomendadas
    cluster = projetion_kmeans.filter(projetion_kmeans.artists_song == nome_musica).select('cluster_pca').collect()[0][0]
    musicas_recomendadas = projetion_kmeans.filter(projetion_kmeans.cluster_pca == cluster)\
                                       .select('artists_song', 'id', 'pca_features')
    componenetes_musica = musicas_recomendadas.filter(musicas_recomendadas.artists_song == nome_musica)\
                                          .select('pca_features').collect()[0][0]

    def calcula_distance(value):
        return euclidean(componenetes_musica, value)

    udf_calcula_distance = f.udf(calcula_distance, FloatType())

    musicas_recomendadas_dist = musicas_recomendadas.withColumn('Dist', udf_calcula_distance('pca_features'))

    recomendadas = spark.createDataFrame(musicas_recomendadas_dist.sort('Dist').take(10)).select(['artists_song', 'id', 'Dist'])

  #Pegar informações da API

    playlist_id = recomendadas.select('id').collect()

    name = []
    url = []
    for i in playlist_id:
        track = sp.track(i[0])
        url.append(track["album"]["images"][1]["url"])
        name.append(track["name"])

  #Plotando capas 

    plt.figure(figsize=(15,10))
    columns = 5
    for i, u in enumerate(url):
        ax = plt.subplot(len(url) // columns + 1, columns, i + 1)
        image = io.imread(u)
        plt.imshow(image)
        ax.get_yaxis().set_visible(False)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(name[i], fontsize = 10)
        plt.tight_layout(h_pad=0.7, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.grid(visible=None)
    plt.show()

# COMMAND ----------

recomendador('M83 - Midnight City')
