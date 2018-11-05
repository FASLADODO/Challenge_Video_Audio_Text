import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

replique = True

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/sequence_text.csv', sep='§')
if replique == False:
    data = data.groupby(['Sequence'])['Text'].sum()
    data = data.reset_index()


stop = set(stopwords.words('french'))
stop.update(['.', ',', '"', "'", '?', '!', ':',
                   ';', '(', ')', '[', ']', '{', '}','-'])


data['Text'] = data['Text'].apply(
    lambda x: ' '.join([i for i in word_tokenize(x) if i not in stop]))


tf = CountVectorizer(max_df=0.95, min_df=2)

tf = pd.DataFrame(tf.fit_transform(data['Text']).todense())

data = pd.concat([data,tf],axis = 'columns')
data.drop(['Text'],axis='columns',inplace=True)
assert data.isnull().sum().sum() == 0
if replique:
    data.to_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tf_replique.csv',index=False,sep='§')
else :
    data.to_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/tf_doc.csv',index=False,sep='§')

