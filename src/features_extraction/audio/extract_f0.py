
import os
from scipy.io.wavfile import read
import pandas as pd
import numpy as np

path_data = 'data/audio/'
path_data = ''

files = [file for file in os.listdir(path) if file.split('.')[-1] == 'wav']

f0_files = [file for file in os.listdir(path) if file.split('.')[-1] == 'f0']

# dict_ = {}

# for filename in files:
#     file = path + filename
#     f, y = read(file)

#     dict_[filename] = np.sum([1/f]*len(y))

# df = pd.DataFrame.from_dict(dict_, orient='index')

# df.columns = ['duration']

# df = df.sort_values('duration', ascending=True)

# df.to_csv("/home/robin/Téléchargement", sep='§')
# print(df.head(5))

