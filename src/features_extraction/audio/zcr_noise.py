"""
PAS FINI
"""


import numpy as np
import matplotlib.pyplot as plt
from vocal_activity_detection import VAD
from librosa.feature import zero_crossing_rate
from extract_f0 import extract_time_speak
from scipy.io.wavfile import read
from nrj import log_energie
from scipy.signal import filtfilt, butter

def extract_zcr(y, sr, speak_windows_nrj, win_len=512, step=256):

    zcr = zero_crossing_rate(y, frame_length=win_len, hop_length=step).reshape(-1)
    time_zcr = np.cumsum([step/sr]*len(zcr))
    plt.plot(zcr)

    time_windows_idx = extract_time_speak(time_zcr, speak_windows_nrj)

    zcr_filt = [np.array(zcr[idx1[1]:idx2[0]]) for idx1, idx2 in zip(time_windows_idx[:-1], time_windows_idx[1:])]
    for t in time_windows_idx:
        plt.axvline(t[0], color='k')
        plt.axvline(t[1], color='r')

    plt.plot([], [], 'k', label='Début de segment')
    plt.plot([], [], 'r', label='Fin de segment')
    plt.legend()
    plt.show()
    # print(zcr_filt)
    # print('=================')
    print(np.mean([np.mean(x) for x in zcr_filt]))

file = 'data/audio/SEQ_005_AUDIO.wav'
sr, sig = read(file)
sig = sig.astype(np.float64)

win_len = 4096
step_len = 2048
nrj, time_nrj = np.array(log_energie(sig, sr, win=win_len, step=step_len))

# plt.plot(time_nrj, nrj)
# plt.show()

b, a = butter(3, 0.15)
nrj_bas = filtfilt(b, a, nrj)

nrj_filt, speak_windows_nrj = VAD(nrj_bas, time_nrj, plot=False)
extract_zcr(sig, sr, speak_windows_nrj)