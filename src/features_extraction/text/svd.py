# Singular Value Decomposition.

# Factorise la matrice a en deux matrices unitaires U et Vh, et un tableau 1-D de valeurs singulières 
# (réelles, non négatives) telles que a == U @ S @ Vh, où S est une matrice de zéros 
# avec une forme appropriée diagonale principale s.

import pandas as pd 
from sklearn.decomposition import TruncatedSVD

N_COMPONENTS = 5

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')
svd = TruncatedSVD(n_components=5)
svd = pd.DataFrame(svd.fit_transform(data.drop(['Sequence'],axis='columns')))
svd = svd.add_prefix(f'Svd_')
svd = pd.concat([data['Sequence'],svd],axis='columns')
svd.to_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/svd_tfidf_{N_COMPONENTS}_punct.csv',index=False,sep='§')
