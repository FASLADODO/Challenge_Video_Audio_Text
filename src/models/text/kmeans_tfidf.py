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

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')

model = KMeans(n_clusters=2,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'],axis = 'columns'))

data['cluster'] = cluster

svd = TruncatedSVD(n_components=2)
svd = pd.DataFrame(svd.fit_transform(data.drop(['Sequence','cluster'],axis='columns')))
svd = svd.add_prefix(f'Svd_')

plt.scatter(svd['Svd_0'], svd['Svd_1'], c=cluster, s=50, cmap='viridis')
plt.show()

