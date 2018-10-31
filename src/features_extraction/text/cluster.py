
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import SpectralClustering


data = pd.read_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/nmf_tfidf_2.csv',sep='§')

models = KMeans(n_clusters=2, random_state=0)

models_predict(data)