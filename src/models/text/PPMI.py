
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

data = pd.read_csv('/home/mickael/Documents/Challenge_Video_Audio_Text/features/text/sequence_text.csv',sep='§')

data = data['Text']
texte = []

for i in range(len(data)):
    texte.append(data)

print(texte)