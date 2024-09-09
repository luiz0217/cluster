import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Coleta de dados
db = pd.read_csv('musicas.csv')

print(db.head())
print(db.info())


# Contar o número musicas por genero
status = db.groupby('genre').agg({
    'track_name': 'count'
}).reset_index()


# Codificar a coluna 'genre' (transformar texto em números)
label_encoder = LabelEncoder()
status['genre_encoded'] = label_encoder.fit_transform(status['genre'])


# Normalizar os dados (apenas a coluna codificada e o número de músicas)
scaler = StandardScaler()
X = scaler.fit_transform(status[['genre_encoded', 'track_name']])




# KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
status['kmeans_cluster'] = kmeans.fit_predict(X)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
status['agg_cluster'] = agg_clustering.fit_predict(X)



# KMeans
silhouette_kmeans = silhouette_score(X, status['kmeans_cluster'])
calinski_kmeans = calinski_harabasz_score(X, status['kmeans_cluster'])

# Agglomerative Clustering
silhouette_agg = silhouette_score(X, status['agg_cluster'])
calinski_agg = calinski_harabasz_score(X, status['agg_cluster'])

print("KMeans Metricas:")
print(f"Qualidade kmeans: {silhouette_kmeans:.4f}")
print(f"Qualidade por Calinski: {calinski_kmeans:.4f}")

print("\nAgglomerative Clustering Metrics:")
print(f"Qualide do Agglomerative: {silhouette_agg:.4f}")
print(f"Qualidade por Calinski: {calinski_agg:.4f}")




# Escolher o melhor algoritmo
if silhouette_kmeans > silhouette_agg and calinski_kmeans > calinski_agg:
    melhor = 'KMeans'
else:
    melhor = 'Agglomerative Clustering'

print(f"\nMelhor: {melhor}")


# Reaplicar o melhor algoritmo
if melhor == 'KMeans':
    final_clusters = status['kmeans_cluster']
else:
    final_clusters = status['agg_cluster']

status['final_cluster'] = final_clusters


# Tratamento de dados - Filtrar apenas as colunas numéricas para agregação
colunas = ['genre_encoded', 'track_name']
cluster_stats = status.groupby('final_cluster')[colunas].agg(['mean', 'std', 'min', 'max'])
print("\nCluster Statistics:")
print(cluster_stats)

# Identificar o genero com mais musicas
maisMusicas = status.loc[status['track_name'].idxmax()]
menosMusicas = status.loc[status['track_name'].idxmin()]

print("\nGenero com mais musicas:")
print(maisMusicas)

print("\nGenero com menos musicas:")
print(menosMusicas)

# Salvar o DataFrame com clusters para referência
status.to_csv('musicas_clustered.csv', index=False)