
# Analyse discriminante linéaire (LDA).

# Un classifieur avec une limite de décision linéaire, généré en ajustant les densités conditionnelles 
# de classe aux données et en utilisant la règle de Bayes.

# Le modèle adapte une densité gaussienne à chaque classe, en supposant que toutes les classes 
# partagent la même matrice de covariance.

# Le modèle ajusté peut également être utilisé pour réduire la dimensionnalité de l'entrée 
# en la projetant dans les directions les plus discriminantes.

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

N_COMPONENTS = 5

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tfidf_doc.csv',sep='§')
lda = LatentDirichletAllocation(n_components=N_COMPONENTS, max_iter=5,random_state=42)
lda = pd.DataFrame(lda.fit_transform(data.drop(['Sequence'],axis='columns')))
lda = lda.add_prefix(f'LDA_')
lda = pd.concat([data['Sequence'],lda],axis='columns')
lda.to_csv(f'/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/lda_tfidf_{N_COMPONENTS}.csv',index=False,sep='§')

