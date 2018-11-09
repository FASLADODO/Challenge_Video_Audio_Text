
import os
from scipy.io.wavfile import read
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import read
from nrj import log_energie

path_data = 'data/audio/'
path_f0 = 'features/audio/f0/'

files = sorted([file for file in os.listdir(path_data) if file.split('.')[-1] == 'wav'])
f0_files = sorted([file for file in os.listdir(path_f0) if file.split('.')[-1] == 'f0'])

f0 = [x for x in pd.read_csv(path_f0 + f0_files[0], sep='\n').values]

# f0 = np.array(f0)[:, 0]
# print(f0.shape)
# b, a = signal.butter(3, 0.05)
# f0_filt = signal.filtfilt(b, a, f0)

# f0_filt = [1 if x > 10 else 0 for x in f0_filt]

time = np.cumsum([0.01]*len(f0))

f, sig = read(path_data + files[0])

fig, ax = plt.subplots(2, 1, figsize=(14, 8))
ax[0].plot(log_energie(sig, f, win=8192, step=4096))
# plt.show()
ax[1].plot(time, f0_filt)
plt.show()
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

