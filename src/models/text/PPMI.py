from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import re
import nltk
from math import log
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.corpus import wordnet
#import nltk
#nltk.download()
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# Plotly imports
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/sequence_text.csv', sep='§')

data = data.groupby(['Sequence'])['Text'].sum() # Découper par séquence ou réplique
data = data.reset_index()

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
            'être's, 'sans', "C'était", 'crois', 'ils', 'Une', 'alors', 'peux', 'faut',"qu'on", 'Alors', 'cette', 
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
#print(data.groupby('Text').apply(' '.join).reset_index())


fenetre = [] # liste des 308 séquences sous forme de fenetre
for i in data['Text']:
    fenetre.append([i])

fenetre = [sentence.split() for f in fenetre for sentence in f] # une fenetre : une séquence
#[f.split() for f in fenetre[0]]
#fenetre

liste = []
for i in data['Text']:
    liste.append(i)
liste = " ".join(liste).split()
liste = [x.lower() for x in liste]

#liste = liste[0:20]
liste = liste[0:100] # on réduit à 300 mots car le corpus est très long 20 000 mots temps long
# print(len(liste))

#proba simple : On calcule la probabilité d'apparition d'un mot dans les fenêtres
def proba_simple(mot, fen):
    count = 0
    for fenetre in fen:
        if mot in fenetre:
            count += 1
    return(count/len(fen))


def proba_conjoint(mot1, mot2, fen):
    count = 0
    for fenetre in fen:
        if mot1 in fenetre:
            if mot2 in fenetre:
                count += 1
    return(count/len(fen))


def proba_conjointe(data):
    freq_2mots = [[0 for i in range(len(liste))] for i in range(len(liste))]
    for mot1 in liste:
        for mot2 in liste:
            if mot1 != mot2:
                den=0
                num=0
                for i in fenetre:
                    if (mot1 or mot2) in i:
                        if (mot1 and mot2) in i:
                            den+=1
                        num +=1   
                if den!= 0:
                     freq_2mots[liste.index(mot1)][liste.index(mot2)]=den/len(liste)
                else:
                     freq_2mots[liste.index(mot1)][liste.index(mot2)]=0
    return freq_2mots

freq_2mots = proba_conjointe(liste)

def proba(data):
    freq_mot = [0 for i in range(len(liste))]
    for mot1 in liste:
        for i in fenetre:
            if mot1 in i:
                freq_mot[liste.index(mot1)]+=1
        freq_mot[liste.index(mot1)] = round(freq_mot[liste.index(mot1)] / len(fenetre),2)
    return freq_mot

freq_mot = proba(liste)

def ppmi(data):
    ppmi = [[0 for i in range(len(liste))] for i in range(len(liste))]
    for mot1 in range(len(liste)):
        for mot2 in range(len(liste)):
            if freq_mot[mot1]==0 or freq_mot[mot2]==0 or freq_2mots[mot1][mot2]==0 or math.log(freq_2mots[mot1][mot2] / (freq_mot[mot1]*freq_mot[mot2])) < 0:

                 ppmi[mot1][mot2]=0
            else:
                ppmi[mot1][mot2]=math.log(freq_2mots[mot1][mot2] / (freq_mot[mot1]*freq_mot[mot2]))
    return ppmi

ppmi_tot = ppmi(liste)

def Calcul(mot1,mot2, freq_2mots, freq_mot, ppmi):
    mot1=liste.index(mot1)
    mot2=liste.index(mot2)
    print("\n Fréquence entre les 2 mots : ",freq_2mots[mot1][mot2],"\n Fréquence du mot 1 :  " ,freq_mot[mot1],"\n Fréquence du mot 2  ",freq_mot[mot2])
    print(" Valeur du ppmi : ", ppmi[mot1][mot2])

print(Calcul('On', 'Ils', freq_2mots, freq_mot, ppmi_tot)) # sur 200 / 20022 mots mais les 308 fenêtres.
# Changer les 2 mots



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.array([ppmi_tot[liste.index(i)] for i in ['']])

pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
red = [0]
bleu = [1]

plt.plot([x[0] for x in Xp[red]],[x[1] for x in Xp[red]], 'ro')
plt.plot([x[0] for x in Xp[bleu]],[x[1] for x in Xp[bleu]], 'bx')

def similarité(liste)  :  
    liste_synonyme = []
    for mot1 in liste:
         if wordnet.synsets(mot1)!=[]:
            liste_synonyme.append(mot1)     
    similarité = [[0 for i in range(len(liste_synonyme))] for i in range(len(liste_synonyme))]
    for mot1 in liste_synonyme:
        for mot2 in liste_synonyme:
                synonyme1 = wordnet.synsets(mot1)[0] # On prendra toujours le premier sens
                synonyme2 = wordnet.synsets(mot2)[0]
                similarité[liste_synonyme.index(mot1)][liste_synonyme.index(mot2)] = synonyme1.path_similarity(synonyme2)
    return similarité

sim = similarité(liste)


