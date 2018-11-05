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

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/emotion_doc.csv',sep='§')


pca = PCA(n_components=2)
pca = pd.DataFrame(pca.fit_transform(data.drop(['Sequence'],axis='columns')))
pca = pca.add_prefix(f'PCA_')

model = KMeans(n_clusters=3,random_state=42,n_init=30)

cluster = model.fit_predict(data.drop(['Sequence'], axis = 'columns'))


plt.scatter(pca['PCA_0'], pca['PCA_1'], c=cluster, s=50, cmap='viridis')
plt.show()

