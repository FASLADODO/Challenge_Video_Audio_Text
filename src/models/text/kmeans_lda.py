

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

N_COMPONENTS = 2
N_CLUSTERS = 2

data = pd.read_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/lda_tfidf_{N_COMPONENTS}.csv',sep='§')

model = KMeans(n_clusters=N_CLUSTERS,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'],axis = 'columns'))

data['cluster'] = cluster

#plt.scatter(data['LDA_0'], data['LDA_1'], c=cluster, s=50, cmap='viridis')
#plt.show()

#plot_cluster(data[['LDA_0','LDA_0']].values, data['Sequence'],cluster, '/home/mickael/Documents/Challenge_Video_Audio_Text/result/plot_cluster_lda.html')
#f.savefig("/home/mickael/Documents/Challenge_Video_Audio_Text/result/kmeans_lda_tf_2.png", bbox_inches='tight')

best_k(data.drop(['Sequence'], axis='columns'), range_min = 20, verbose = True)
