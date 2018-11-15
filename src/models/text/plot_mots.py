import codecs
#import nltk
#nltk.download()
import collections
import math
import re
from math import log
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from scipy.misc import imread
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import plotly.graph_objs as go
# Plotly imports
import plotly.offline as py
import plotly.tools as tls
from wordcloud import STOPWORDS, WordCloud

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/sequence_text.csv', sep='§', engine='python')

data = data.groupby(['Sequence'])['Text'].sum() # Découper par séquence ou réplique
data = data.reset_index()

data['Text'] = [x.lower() for x in data['Text']]

#data['Text'] = [x.lower() for x in data['Text']]
stop = set(stopwords.words('french'))
stop.update(['.', ',', '"', "'", '?', '!', ':',
                   ';', '(', ')', '[', ']', '{', '}','-','...', '..', '«', '»' ,"'", "’", "``", "''",
            'le', 'la', 'les', 'un', 'une', 'des', 'or', 'ni', 'car', 'assez', 'aussitôt', 'assez', 
             'car', 'des', 'la', 'le', 'les', 'ni', 'or', 'un', 'une',
            'a', "C'est", 'Vous', 'ça', "c'est", 'Et', 'Plus', 'Mais', 'Tout', "qu'il",
            'Le', 'quoi', "qu'il", 'si', 'là', 'Ah', 'sais', 'rien', "j'ai", 'Ça', 'Ce', 'Si', 'deux', 'peu',
            'Cette', 'moi', 'oh', 'Les', 'En', 'Oh', 'A', 'Ca', 'Un', 'Va','va', 'C', "m'a", 'vais', 'Vais',
            'un', 'Un', 'La', 'aussi', 'très', 'tout', 'plus', 'alors', 'faut', 'trois', 'dire', 'faire',
            'être', 'sans', "C'était", 'crois', 'ils', 'Une', 'alors', 'peux', 'faut',"qu'on", 'Alors', 'cette', 
            'dit', "d'un", 'dis', 'Il', 'On', 'on', 'tu', 'Tu', "J'ai", "j'ai", 'J', 'Quand', 'quand', 'elle',
            'tous', 'allez', 'moi', 'voilà', 'cent', 'ou', 'pour', 'hein', 'ils', "j'en", "qu'est-ce", 'mon',
            'ans','fois', 'avec', 'fais','avec', "n'est", 'nous', 'vous', 'fait', 'moi', 'ah', 'puis', 'tant',
            'autre', "t'es", 'juste', 'peut', 'tu', 'où', "t'a", "t'as", "c'est", 'autre', 'bah', 'après',
            'comme', "qu'elle", "c'était", 'ben', 'veut', 'je', "s'est", 'ca', 'quelle', "qu'elle", 'chez',
            "d'une", "d'un", 'ouais', 'ouai', 'tu', 'trop', 'veut', 'prend', "d'être", 'parce', 'vous', 'nous',
            'toute', 'quatre', 'se', 'ce', 'tout', 'toute', 'parce', 'huit', 'quel', 'déjà', 'dix', 'huit',
            'cela', 'Le', 'nom', 'six', "l'a", "qu'il", 'coté', "l'ai", "qu'un", "n'a", 'cet', 'cette', 'pour'])


data['Text'] = data['Text'].apply(
    lambda x: ' '.join([i for i in word_tokenize(x) if i not in stop])) 
# data.iloc[5] data = [data['Text']]
# print(data.groupby('Text').apply(' '.join).reset_index())

liste = []
for i in data['Text']:
    liste.append(i)
liste = " ".join(liste).split()
liste = [x.lower() for x in liste]

all_words = data['Text'].str.split(expand=True).unstack().value_counts()

data = [go.Bar(
            x = all_words.index.values[1:50],
            y = all_words.values[1:50],
            marker= dict(colorscale='Jet',
                         color = all_words.values[1:100]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Top 50 Word frequencies in the dataset'
)

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='/home/mickael/Documents/Challenge_Video_Audio_Text/result/Top 50 Word frequencies.html')