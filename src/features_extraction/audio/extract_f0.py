
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from nrj import log_energie
from vocal_activity_detection import VAD
from scipy.signal import filtfilt, butter
from Yin import yin
import pandas as pd

def find_idx_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def extract_time_speak(time, speak_windows_nrj):
    """
    Fonction permettant de transposer les zones de paroles 
    (calculées à partir de l'énergie) à une autre forme (F0, signal initial...)   
    
    Arguments:
        time {numpy array} -- Liste contenant le temps (en secondes) 
        associé à chaque point de la fréquence fondamentale
        
        speak_windows_nrj {numpy array} -- tableau contenant les fenêtres de paroles 
        (en secondes) calculées sur l'énergie
        (de la forme : [[debut_1, fin_1], [debut_2, fin_2], ...] avec debut_i (ou fin_i) le temps
        (secondes) correspondant au début (ou fin) de la ième fenêtre de parole)
    
    Returns:
        {numpy array} -- un tableau de la même forme que speak_windows_nrj mais contenant les index
        (et non le temps lui-même) du tableau time
    """

    speak_windows_f0 = []
    for t in speak_windows_nrj:
        start = t[0]
        end = t[1]

        speak_windows_f0.append([find_idx_nearest(time, t[0]), find_idx_nearest(time, t[1])])
    return speak_windows_f0


def extract_f0(wav_file, path_f0='features/audio/f0/', use_yin=True):
    """
    Extraction de la fréquence fondamentale d'un signal audio en la calculant
    uniquement sur les zones de paroles. Cette fréquence est normalisée pour chaque 
    zone de parole (division par la moyene de la F0 de la zone de parole)
    
    Arguments:
        wav_file {[str]} -- fichier .wav du signal
    
    Keyword Arguments:
        path_f0 {str} -- dossier où sont stockés les fichier contenant la fréquence
            fondamentale calculée par l'algorithme AMDF (si use_yin = False)   
            (default: {'features/audio/f0/'})
        use_yin {bool} -- Booleen, mettre True si vous voulez utiliser l'algorithme yin
                          Mettre False pour utiliser l'algorithme AMDF (default: {True})
    
    Returns:
        {numpy array} -- tableau à une dimension contenant la fréquence fondamentale 
                         (normalisée) sur les zones de paroles
    """

    if not use_yin:
        file_f0 = path_f0 + wav_file.split('/')[-1].split('.')[0] + '.f0'

    sr, sig = read(wav_file)

    win_len = 4096
    step_len = 2048
    nrj, time_nrj = log_energie(sig, sr, win=win_len, step=step_len)

    nrj = np.array(nrj)
    time_nrj = np.array(time_nrj)

    b, a = butter(3, 0.15)
    nrj_bas = filtfilt(b, a, nrj)

    nrj_filt, speak_windows_nrj = VAD(nrj_bas, time_nrj, plot=False)

    if use_yin:
        f0, _, _, time_f0 = yin.compute_yin(sig, sr, w_len=512, w_step=256, f0_min=90, f0_max=300, harmo_thresh=0.6)
    else:
        f0 = [float(x) for x in pd.read_csv(file_f0, sep='\n').values]
        time_f0 = np.cumsum([0.01]*len(f0))

    speak_windows_f0 = extract_time_speak(time_f0, speak_windows_nrj)

    f0_speak = [np.array(f0[idx[0]:idx[1]]) for idx in speak_windows_f0]

    list_freq = []
    for freq in f0_speak:
        freq = freq[freq != 0]
        if len(freq) > 0:
            freq = np.log10(freq/np.mean(freq))
            list_freq.extend(freq)

    return list_freq

if __name__ == '__main__':

    file = 'data/audio/SEQ_001_AUDIO.wav'
    extract_f0(file)

    file = 'data/audio/SEQ_002_AUDIO.wav'
    extract_f0(file)

    file = 'data/audio/SEQ_003_AUDIO.wav'
    extract_f0(file)

    file = 'data/audio/SEQ_004_AUDIO.wav'
    extract_f0(file)

    file = 'data/audio/SEQ_016_AUDIO.wav'
    extract_f0(file)
# plt.plot(extract_f0(file), 'rx')
# plt.show()





