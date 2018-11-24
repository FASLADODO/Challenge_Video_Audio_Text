

""" 
Kmeans : Le partitionnement en k-moyennes (ou k-means en anglais) est une méthode de partitionnement 
de données et un problème d'optimisation combinatoire. Étant donnés des points et un entier k, 
le problème est de diviser les points en k groupes, souvent appelés clusters, de façon 
à minimiser une certaine fonction. On considère la distance d'un point à la moyenne 
des points de son cluster ; la fonction à minimiser est la somme des carrés de ces distances. 
"""


from plotly.offline import download_plotlyjs, plot, iplot
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from plt import plot_cluster
from evaluation_cluster import best_k

N_CLUSTERS = 5

data = pd.read_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')

model = KMeans(n_clusters=N_CLUSTERS,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'],axis = 'columns'))

data['cluster'] = cluster

cluster = pd.Series(cluster)
cluster.name = 'Cluster'

svd = TruncatedSVD(n_components=2) # dimension trop élevé, du coup, on réduit !
svd = pd.DataFrame(svd.fit_transform(data.drop(['Sequence','cluster'],axis='columns')))
svd = svd.add_prefix(f'Svd_')

#plt.scatter(svd['Svd_0'], svd['Svd_1'], c=cluster, s=50, cmap='viridis')
#plt.show()

plot_cluster(svd[['Svd_0','Svd_1']].values, data['Sequence'],cluster.values,'/home/mickael/Documents/Challenge_Video_Audio_Text/result/plot_cluster_tfidf_5.html')
#f.savefig("/home/mickael/Documents/Challenge_Video_Audio_Text/result/kmeans_tfidf_2.png", bbox_inches='tight')

""" Calcul du score avec silhouette // k=2 le mieux pour chaque """
#best_k(data.drop(['Sequence'], axis='columns'), range_min = 20, verbose = True)


# print(data.head())

# ann = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/data/external/annotation.csv')
# ann = ann[['Sequence','Violent']]
# print(ann.head())
# print(ann.shape)

# val = pd.concat([data[['Sequence','cluster']],ann],axis = 'columns')
# val  = val.dropna()

# print(accuracy_score(val['cluster'],val['Violent']))
