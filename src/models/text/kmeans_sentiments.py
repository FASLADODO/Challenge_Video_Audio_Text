
# https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-explo-acp.pdf

# L'analyse en composantes principales (ACP ou PCA en anglais pour principal component analysis), est une méthode 
# de la famille de l'analyse des données et plus généralement de la statistique multivariée, 
# qui consiste à transformer des variables liées entre elles (dites « corrélées » en statistique) 
# en nouvelles variables décorrélées les unes des autres. Ces nouvelles variables sont nommées 
# « composantes principales », ou axes principaux. Elle permet au praticien de réduire 
# le nombre de variables et de rendre l'information moins redondante. 

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import warnings
warnings.filterwarnings('ignore')
from plt import plot_cluster


data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/emotion_doc.csv',sep='§')

pca = PCA(n_components=2)
pca = pd.DataFrame(pca.fit_transform(data.drop(['Sequence'],axis='columns')))
pca = pca.add_prefix(f'PCA_')

model = KMeans(n_clusters=3,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'], axis = 'columns'))

plt.scatter(pca['PCA_0'], pca['PCA_1'], c=cluster, s=50, cmap='viridis')
plt.show()

plot_cluster(pca[['PCA_0','PCA_1']].values, data['Sequence'],cluster, '/home/mickael/Documents/Challenge_Video_Audio_Text/result/plot_cluster_sentiments.html')
#f.savefig("/home/mickael/Documents/Challenge_Video_Audio_Text/result/kmeans_sentiments.png", bbox_inches='tight')

