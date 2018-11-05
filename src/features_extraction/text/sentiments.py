#

import pandas as pd 
from pyFeel import Feel

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/sequence_text.csv',sep='§')

replique = False

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/sequence_text.csv', sep='§')
if replique == False:
    data = data.groupby(['Sequence'])['Text'].sum()
    data = data.reset_index()
series_emotion = data['Text'].apply(lambda x : Feel(x).emotions())
emo = pd.DataFrame.from_records(series_emotion.values.tolist() )
data = pd.concat([data.drop(['Text'],axis='columns'),emo],axis ='columns')

if replique:
    data.to_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/emotion_replique.csv',index=False,sep='§')
else :
    data.to_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/emotion_doc.csv',index=False,sep='§')
