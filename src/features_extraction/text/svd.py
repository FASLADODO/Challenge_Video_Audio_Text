
import pandas as pd 
from sklearn.decomposition import TruncatedSVD

N_COMPONENTS = 2

data = pd.read_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/tfidf_doc.csv',sep='§')
svd = TruncatedSVD(n_components=5)
svd = pd.DataFrame(svd.fit_transform(data.drop(['Sequence'],axis='columns')))
svd = svd.add_prefix(f'Svd_')
svd = pd.concat([data['Sequence'],svd],axis='columns')
svd.to_csv(f'/home/mickael/Documents/challenge_son_vidéo_texte/features/text/svd_tfidf_{N_COMPONENTS}.csv',index=False,sep='§')
