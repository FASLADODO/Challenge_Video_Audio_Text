

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
from sklearn.manifold import TSNE

N_COMPONENTS = 2
N_CLUSTERS = 5

data = pd.read_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/nmf_tfidf_{N_COMPONENTS}_TSNE.csv',sep='§')

model = KMeans(n_clusters=N_CLUSTERS,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'],axis = 'columns'))

data['cluster'] = cluster

#plt.scatter(data['nmf_0'], data['nmf_1'], c=cluster, s=50, cmap='viridis')
#plt.show()

plot_cluster(data[['nmf_0','nmf_1']].values, data['Sequence'],cluster, '/home/mickael/Documents/Challenge_Video_Audio_Text/result/plot_cluster_nmf_5_TSNE.html')


#best_k(data.drop(['Sequence'], axis='columns'), range_min = 20, verbose = True)
