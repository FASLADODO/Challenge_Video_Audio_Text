
import pandas as pd
from sklearn.decomposition import NMF

N_COMPONENTS = 2

data = pd.read_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/tfidf_doc.csv',sep='§')
nmf = NMF(n_components=N_COMPONENTS)
nmf = pd.DataFrame(nmf.fit_transform(data.drop(['Sequence'],axis='columns')))
nmf = nmf.add_prefix(f'nmf_')
nmf = pd.concat([data['Sequence'],nmf],axis='columns')
nmf.to_csv(f'/home/mickael/Documents/challenge_son_vidéo_texte/features/text/nmf_tfidf_{N_COMPONENTS}.csv',index=False,sep='§')

