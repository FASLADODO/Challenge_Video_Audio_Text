import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plot_cluster import plot_cluster
from clustering import clustering
from sklearn.preprocessing import normalize

path_csv = 'features/merge/'


emotion = pd.read_csv(path_csv + 'emotion_doc.csv', sep='§', index_col='Sequence', engine='python')
nmf = pd.read_csv(path_csv + 'nmf_tfidf_5.csv', sep='§', index_col='Sequence', engine='python')

text = pd.merge(emotion, nmf, right_index=True, left_index=True)
text_norm = pd.DataFrame(normalize(text), index=text.index, columns=text.columns)
print(text_norm.head(1))


audio1 = pd.read_csv(path_csv + 'locutors.csv', sep='§', index_col='Sequence', engine='python')
audio2 = pd.read_csv(path_csv + 'audio_nrj_f0.csv', sep='§', index_col='Sequence', engine='python')
audio2 = audio2.drop([col for col in audio2.columns if 'skew' in col or 'kurt' in col], axis=1)
audio = pd.merge(audio1, audio2, right_index=True, left_index=True)

audio_norm = pd.DataFrame(normalize(audio), index=audio.index, columns=audio.columns)
print(audio_norm.head(1))

pca = PCA(n_components=50)

video = pd.read_csv(path_csv + 'df_histo.csv', sep='§', index_col=0, engine='python')
video.index = [idx[:7] for idx in video.index]
video_norm = pd.DataFrame(pca.fit_transform(normalize(video)), index=video.index)
print(video_norm.head(1))
print(sum(pca.explained_variance_ratio_))

path = 'result/merge/'

audio_video = pd.merge(video_norm, audio_norm, right_index=True, left_index=True)
clustering(audio_video, path + 'audio_video.html', nb_cluster=1)

audio_text = pd.merge(audio_norm, text_norm, right_index=True, left_index=True)
clustering(audio_text, path + 'audio_text.html', nb_cluster=1)

video_text = pd.merge(video_norm, text_norm, right_index=True, left_index=True)
clustering(video_text, path + 'video_text.html', nb_cluster=1)

audio_video_text = pd.merge(video_text, audio_norm, right_index=True, left_index=True)
audio_video_text.to_csv('features/merge/all.csv')
# audio_video_text_norm = pd.DataFrame(normalize(audio_video_text), index=audio_video_text.index, columns=audio_video_text.columns)
clustering(audio_video_text, path + 'audio_video_text.html', nb_cluster=1)
