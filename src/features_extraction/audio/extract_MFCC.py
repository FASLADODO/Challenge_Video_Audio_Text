import ast
import librosa
import os
import pandas as pd
from scipy.io.wavfile import read
from tqdm import tqdm
from collections import defaultdict
from nrj import log_energie
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
from vocal_activity_detection import HOS, VAD
from parse_LIUM import features_LIUM
from plot_cluster import plot_cluster

path = 'data/audio/'

files = sorted([file for file in os.listdir(path) if file.split('.')[-1] == 'wav'])

dict_features = {}

for filename in tqdm(files):
    file = path + filename

    name_seq = filename.split('.')[0]

    sr, sig = read(file)
    sig = sig.astype(np.float64)
    # sig /= np.linalg.norm(sig)

    win_len = 4096
    step_len = 2048
    nrj = log_energie(sig, sr, win=win_len, step=step_len)
    # plt.plot(nrj)
    # plt.show()
    nrj_filt = VAD(nrj, plot=False)
    nrj_filt = np.array(nrj_filt)

    ratio_speak = np.count_nonzero(np.isnan(nrj_filt)) / len(nrj_filt)

    filter_ = [not(x) for x in np.isnan(nrj_filt)]

    nrj_filt = nrj_filt[filter_]

    nrj_filt /= np.linalg.norm(nrj_filt)

    mean_nrj, std_nrj, skew_nrj, kurt_nrj = HOS(nrj_filt)
    # MFCC = librosa.feature.mfcc(sig, sr, n_mfcc=13)

    # dict_features[filename].extend(np.mean(MFCC, axis=1))
    # dict_features[filename].extend(np.std(MFCC, axis=1))
    
    # nrj = np.log10(np.sum(np.power(sig, 2)))
    # zcr = librosa.feature.zero_crossing_rate(sig, frame_length=2048, hop_length=512).sum()
    
    dict_features[name_seq] = {'ratio_speak':ratio_speak,
                               'mean_nrj':mean_nrj, 
                               'std_nrj': std_nrj,
                               'skewness_nrj': skew_nrj,
                               'kurtosis_nrj': kurt_nrj}

df = pd.DataFrame.from_dict(dict_features, orient='index')
df = pd.merge(df, features_LIUM('features/audio/LIUM_segmentation/'), left_index=True, right_index=True)

print(df.head(6))

y = pd.read_csv('data/external/Annotations.csv', sep='§')
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)


kmeans = sklearn.cluster.KMeans(n_clusters=2)
labels = kmeans.fit_predict(df)

pca = sklearn.decomposition.PCA(n_components=2)
transf = pd.DataFrame(pca.fit_transform(df), index=df.index)


plot_cluster(transf.values, transf.index, labels, 'cluster_audio.html')
## print(type(transf))
## print(transf.shape)

## plt.scatter(transf[:, 0], transf[:, 1], c=labels)
## plt.show()

## print(df.head(5))


## import plotly.graph_objs as go
## from plotly.offline import download_plotlyjs, plot, iplot



##print(dict_features[filename])