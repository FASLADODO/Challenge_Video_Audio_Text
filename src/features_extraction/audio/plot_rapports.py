import ast
import librosa
import os
import pandas as pd
from scipy.io.wavfile import read
from tqdm import tqdm
from collections import defaultdict
from nrj import log_energie
import numpy as np
from extract_f0 import extract_f0
import matplotlib.pyplot as plt
from vocal_activity_detection import HOS, VAD
from parse_LIUM import features_LIUM
from scipy.signal import filtfilt, butter

path = 'data/audio/'

files = sorted([file for file in os.listdir(path) if file.split('.')[-1] == 'wav'])

filename = files[1]

file = path + filename

# sr, sig = read(file)
# sig = sig.astype(np.float64)
# sig /= np.linalg.norm(sig)
# time_sig = np.cumsum([1/sr]*len(sig))


# f0 = extract_f0(file, use_yin=False)
# mean_f0, std_f0, skew_f0, kurt_f0 = HOS(f0)

# win_len = 2048
# step_len = 1024
# nrj, time_nrj = np.array(log_energie(sig, sr, win=win_len, step=step_len))

# fig, ax = plt.subplots(2, 1, figsize=(14, 8))
# ax[0].plot(time_sig, sig)
# ax[0].set_title('Signal Initial')
# ax[1].plot(time_nrj, nrj)
# ax[1].set_title('Log énergie')
# ax[1].set_xlabel('Temps (secondes)')
# plt.show()
# plt.plot(time_nrj, nrj)
# plt.show()

# b, a = butter(3, 0.15)
# nrj_bas = filtfilt(b, a, nrj)

# path_csv = 'features/merge/'

# lium = pd.read_csv(path_csv + 'locutors.csv', sep='§', index_col='Sequence', engine='python')
# print(set(lium['nb_locutors']))
# plt.hist(lium['nb_locutors'], bins=50, rwidth=10)
# plt.xlabel('Nombre de locuteurs détectés')
# plt.ylabel('Nombre de films associés')
# plt.show()

