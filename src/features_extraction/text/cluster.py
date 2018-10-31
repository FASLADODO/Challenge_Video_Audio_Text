
import pandas as pd
from sklearn.cluster import SpectralClustering

data = pd.read_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/nmf_tfidf_2.csv',sep='§')
sc = SpectralClustering(3, affinity='precomputed', n_init=100,
                        assign_labels='discretize')

sc.fit_predict(data)