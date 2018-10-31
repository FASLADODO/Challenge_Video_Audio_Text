
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

replique = False

data = pd.read_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/sequence_text.csv', sep='§')
if replique == False:
    data = data.groupby(['Sequence'])['Text'].sum()
    data = data.reset_index()


stop = set(stopwords.words('french'))
stop.update(['.', ',', '"', "'", '?', '!', ':',
                   ';', '(', ')', '[', ']', '{', '}','-'])


data['Text'] = data['Text'].apply(
    lambda x: ' '.join([i for i in word_tokenize(x) if i not in stop]))


tfidf = TfidfVectorizer(max_df=0.95, min_df=2)

tfidf = pd.DataFrame(tfidf.fit_transform(data['Text']).todense())

data = pd.concat([data,tfidf],axis = 'columns')
data.drop(['Text'],axis='columns',inplace=True)
assert data.isnull().sum().sum() == 0
if replique:
    data.to_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/tfidf_replique.csv',index=False,sep='§')
else: 
    data.to_csv('/home/mickael/Documents/challenge_son_vidéo_texte/features/text/tfidf_doc.csv',index=False,sep='§')