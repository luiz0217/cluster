import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

status = pd.read_csv('musicas_clustered.csv')

sns.set(style='darkgrid')

plt.figure(figsize=(16,8))

#kmeans grafico
plt.subplot(1,2,1)
sns.scatterplot(data=status, x='genre',y='track_name',hue='kmeans_cluster',palette = 'viridis',s=100, edgecolor='w')
plt.title('KMeans')
plt.xlabel('genero')
plt.ylabel('musicas')
plt.legend(title='cluster',loc='best')

#Agglomerative Clustering grafico
plt.subplot(1,2,2)
sns.scatterplot(data=status, x='genre',y='track_name',hue='agg_cluster',palette = 'viridis',s=100, edgecolor='w')
plt.title('Agglomerative')
plt.xlabel('genero')
plt.ylabel('musicas')
plt.legend(title='cluster',loc='best')

plt.tight_layout
plt.show()