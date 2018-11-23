# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


path_csv = 'features/merge/'

emotion = pd.read_csv(path_csv + 'emotion_doc.csv', sep='§', index_col='Sequence', engine='python')
nmf = pd.read_csv(path_csv + 'nmf_tfidf_5.csv', sep='§', index_col='Sequence', engine='python')
text = pd.merge(emotion, nmf, right_index=True, left_index=True)
text_norm = pd.DataFrame(normalize(text), index=text.index, columns=text.columns)
# print(text_norm.head(1))


audio1 = pd.read_csv(path_csv + 'locutors.csv', sep='§', index_col='Sequence', engine='python')
audio2 = pd.read_csv(path_csv + 'audio_nrj_f0.csv', sep='§', index_col='Sequence', engine='python')
# audio2 = audio2.drop([col for col in audio2.columns if 'skew' in col or 'kurt' in col], axis=1)
audio = pd.merge(audio1, audio2, right_index=True, left_index=True)

audio_norm = pd.DataFrame(normalize(audio), index=audio.index, columns=audio.columns)


pca = PCA(n_components=50)

video1 = pd.read_csv(path_csv + 'df_histo.csv', sep='§', index_col=0, engine='python')
video1.index = [idx[:7] for idx in video1.index]
video2 = pd.read_csv(path_csv + 'df_cuts.csv', sep='§', index_col=0, engine='python')
video2.index = [idx[:7] for idx in video2.index]
video3 = pd.read_csv(path_csv + 'df_momentum.csv', sep='§', index_col=0, engine='python')
video3.index = [idx[:7] for idx in video3.index]
video = pd.merge(video1, video2, right_index=True, left_index=True)
video = pd.merge(video, video3, right_index=True, left_index=True)
video.index = [idx[:7] for idx in video.index]
video_norm = pd.DataFrame(pca.fit_transform(normalize(video)), index=video.index)
video_norm.index.name = 'Sequence'
video_norm = video_norm.add_prefix('PCA_')
# df = pd.merge(text_norm, video_norm, right_index=True, left_index=True)
df = pd.merge(audio_norm, video_norm, right_index=True, left_index=True)
df.drop(['nb_locutors', 'ratio_speak', 'ratio_HF'], axis=1, inplace =True)
# df = pd.merge(audio_norm, text_norm, right_index=True, left_index=True)


# df = video_norm
# df = audio_norm
# df = text_norm
# df = pd.read_csv('features/merge/all.csv', sep='§', index_col='Sequence', engine='python')


path_annot = 'annotations/'
objectif = 'Exterieur' # Exterieur

y_train = pd.read_csv(path_annot + f'y_train_{objectif.lower()}.csv', sep='§', index_col='Sequence', usecols=[objectif, 'Sequence'], engine='python')
y_test = pd.read_csv(path_annot + f'y_test_{objectif.lower()}.csv', sep='§', index_col='Sequence', usecols=[objectif, 'Sequence'], engine='python')

print(len(y_train.values[y_train.values == 0]) / y_train.shape[0])

X_train = pd.merge(df, y_train, right_index=True, left_index=True)
X_train.drop([objectif], axis=1, inplace =True)
X_test = pd.merge(df, y_test, right_index=True, left_index=True)
X_test.drop([objectif], axis=1, inplace =True)

print(X_train.head(5))

lgbm = LGBMClassifier(n_estimators=1000, num_leaves=31)

lgbm.fit(X_train, y_train.values.reshape(-1))

y_pred = lgbm.predict(X_test)
print(accuracy_score(y_test.values.reshape(-1), y_pred))

feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

# y_test = 
