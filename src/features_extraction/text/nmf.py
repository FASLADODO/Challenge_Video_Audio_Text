
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE



data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')
nmf = TSNE(n_components=N_COMPONENTS)
nmf = pd.DataFrame(nmf.fit_transform(data.drop(['Sequence'],axis='columns')))
nmf = nmf.add_prefix(f'nmf_')
nmf = pd.concat([data['Sequence'],nmf],axis='columns')
nmf.to_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/nmf_tfidf_{N_COMPONENTS}_TSNE.csv',index=False,sep='§')

