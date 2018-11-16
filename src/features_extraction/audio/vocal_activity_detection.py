
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.io.wavfile import read
from nrj import log_energie
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, butter
from scipy.stats import skew, kurtosis

def HOS(X):
    try:
        mean = np.mean(X)
        std = np.std(X)
        skewness = skew(X)
        kurt = kurtosis(X)
    except:
        print(X)
    return mean, std, skewness, kurt

def gaussian(x, mu, sig):
        return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def VAD(nrj, time_nrj, nrj_n, plot=False):

    if type(nrj) == list:
        nrj = np.array(nrj)
        
    gmm = GaussianMixture(n_components=2, covariance_type='diag')

    gmm.fit(nrj.reshape(len(nrj), 1))

    moy = gmm.means_
    sig = gmm.covariances_
    
    search_grid = np.linspace(min(moy), max(moy), 100)

    g1 = gaussian(search_grid, moy[0], sig[0])
    g2 = gaussian(search_grid, moy[1], sig[1])

    # print(min(abs(moy)), max(abs(moy)))
    diff = abs(g1 - g2)

    thresh = search_grid[np.argmin(diff)]
    
    list_times = []
    i = 0
    while i < len(nrj) - 1:
        sub_list = []
        if nrj[i] > thresh:
            sub_list.append(time_nrj[i])
            j = 1
            while (nrj[i + j] > thresh) and ((i + j + 1) < len(nrj)):
                j += 1
            sub_list.append(time_nrj[i + j])
            i = i + j
            list_times.append(sub_list)
        else:
            i += 1

    if plot:
        x = np.linspace(min(moy) - 3*sig[np.argmin(moy)], max(moy) + 3*sig[np.argmax(moy)], 1000)
        # x = np.linspace(0, 0.2, 100)
        
        g3 = gaussian(x, moy[0], sig[0])
        g4 = gaussian(x, moy[1], sig[1])

        fig, ax = plt.subplots(3, 1, figsize=(14, 12))
        ax[0].plot(time_nrj, nrj_n, 'r', label='Énergie')
        ax[0].plot(time_nrj, nrj, 'b', label='Énergie filtrée')
        ax[0].set_xlabel('Temps (s)')
        ax[0].set_ylim(-5.5, -1)
        ax[0].legend(loc='upper left')
        ax[1].plot(x, g3, label='Parole')
        ax[1].plot(x, g4, label='Bruit')
        ax[1].axvline(thresh, color='k', label='Seuil')
        ax[1].legend(loc='upper left')
        ax[1].set_xlabel('Energie')
        ax[2].plot(time_nrj, nrj)
        ax[2].axhline(thresh, color='k', label='Seuil')
        for t in list_times:
            ax[2].fill_between(np.linspace(t[0], t[1], 100), y1=-7, y2=-1, color='g', alpha=0.2)
        ax[2].fill_between([0,0], y1=-7, y2=-1, color='g', alpha=0.2, label='Paroles')
        ax[2].legend(loc='upper left')
        ax[2].set_xlabel('Temps (s)')
        ax[2].set_ylim(-5.5, -1)
        plt.subplots_adjust(hspace=0.3)
        plt.show()

    nrj_filt = np.array([x if x > thresh else np.NaN for x in nrj]) 
    return nrj_filt, list_times

if __name__ == '__main__':
    file = 'data/audio/SEQ_002_AUDIO.wav'

    f, y = read(file)

    y = y.astype(np.float64)
    y /= np.linalg.norm(y)

    win = 2048
    step = 1024

    nrj, time_nrj = log_energie(y, f, win=win, step=step)
    nrj = np.array(nrj)
    nrj_2 = nrj
    b, a = butter(3, 0.19)
    nrj = filtfilt(b, a, nrj)

    # nrj = nrj / np.mean(nrj)

    nrj_filt, time = VAD(nrj, time_nrj, nrj_2, plot=True)
    
    plt.figure(1, figsize=(14, 4))
    plt.plot(time_nrj, nrj_2, label='Énergie')
    for t in time:
        # plt.axvline(t[0], color='k')
        # plt.axvline(t[1], color='r')
        plt.fill_between(np.linspace(t[0], t[1], 100), y1=-7, y2=-1, color='g', alpha=0.2)
    plt.fill_between([0,0], y1=-7, y2=-1, color='g', alpha=0.2, label='Paroles')
    # plt.plot([], [], 'k', label='Début de segment')
    # plt.plot([], [], 'r', label='Fin de segment')
    plt.legend(loc='lower right')
    plt.show()
    # print(time)
