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

dict_features = {}

for filename in tqdm(files):
    file = path + filename

    name_seq = filename.split('.')[0][:7]

    sr, sig = read(file)
    sig = sig.astype(np.float64)
    sig /= np.linalg.norm(sig)
    
    f0 = extract_f0(file, use_yin=False)
    mean_f0, std_f0, skew_f0, kurt_f0 = HOS(f0)

    win_len = 4096
    step_len = 2048
    nrj, time_nrj = np.array(log_energie(sig, sr, win=win_len, step=step_len))
    

    b, a = butter(3, 0.15)
    nrj_bas = filtfilt(b, a, nrj)

    nrj_filt, _ = VAD(nrj_bas, time_nrj, plot=False)

    ratio_speak = np.count_nonzero(np.isnan(nrj_filt)) / len(nrj_filt)
    filter_ = [not(x) for x in np.isnan(nrj_filt)]
    nrj_filt = nrj[filter_]

    mean_nrj, std_nrj, skew_nrj, kurt_nrj = HOS(nrj_filt)

    dict_features[name_seq] = { 'ratio_speak':ratio_speak,
                               
                                'mean_nrj':mean_nrj, 
                                'std_nrj': std_nrj,
                                'skewness_nrj': skew_nrj,
                                'kurtosis_nrj': kurt_nrj,

                                'mean_f0':mean_f0, 
                                'std_f0': std_f0,
                                'skewness_f0': skew_f0,
                                'kurtosis_f0': kurt_f0
                            }

df = pd.DataFrame.from_dict(dict_features, orient='index')

df.to_csv('features/audio/csv/audio_nrj_f0.csv', sep='§', index_label='Sequence')
df.head(5)
df = pd.merge(df, features_LIUM('features/audio/LIUM_segmentation/'), left_index=True, right_index=True)
df.to_csv('features/audio/csv/audio_all.csv', sep='§', index_label='Sequence')