# Factorisation de matrice non-négative (NMF)

# Trouvez deux matrices non négatives (W, H) dont le produit se rapproche de la matrice X non négative. 
# Cette factorisation peut être utilisée par exemple pour la réduction de la dimensionnalité, 
# la séparation de source ou l'extraction de sujets.

import pandas as pd
from sklearn.decomposition import NMF

N_COMPONENTS = 5

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')
nmf = NMF(n_components=N_COMPONENTS)
nmf = pd.DataFrame(nmf.fit_transform(data.drop(['Sequence'],axis='columns')))
nmf = nmf.add_prefix(f'nmf_')
nmf = pd.concat([data['Sequence'],nmf],axis='columns')
nmf.to_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/nmf_tfidf_{N_COMPONENTS}.csv',index=False,sep='§')

