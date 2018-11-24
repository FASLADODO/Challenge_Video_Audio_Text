

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE


N_COMPONENTS = 5

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')
lda = LatentDirichletAllocation(n_components=N_COMPONENTS, max_iter=5,random_state=42)
lda = pd.DataFrame(lda.fit_transform(data.drop(['Sequence'],axis='columns')))
lda = lda.add_prefix(f'LDA_')
lda = pd.concat([data['Sequence'],lda],axis='columns')
lda.to_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/lda_tfidf_{N_COMPONENTS}_TSNE.csv',index=False,sep='§')

