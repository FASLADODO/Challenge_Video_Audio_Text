import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('features/text/svd_tfidf_2.csv',sep='§')

model = KMeans(n_clusters=2,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'],axis = 'columns'))

data['cluster'] = cluster


plt.scatter(data['Svd_0'], data['Svd_1'], c=cluster, s=50, cmap='viridis')
plt.show()

