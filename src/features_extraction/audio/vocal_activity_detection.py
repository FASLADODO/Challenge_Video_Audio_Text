
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.io.wavfile import read
from nrj import log_energie
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

def HOS(X):
    mean = np.mean(X)
    std = np.std(X)
    skewness = skew(X)
    kurt = kurtosis(X)
    return mean, std, skewness, kurt

def gaussian(x, mu, sig):
        return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def VAD(nrj, plot=False):

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
    
    if plot:
        x = np.linspace(min(moy) - 3*sig[np.argmin(moy)], max(moy) + 3*sig[np.argmax(moy)], 1000)
        # x = np.linspace(0, 0.2, 100)
        
        g3 = gaussian(x, moy[0], sig[0])
        g4 = gaussian(x, moy[1], sig[1])

        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(x, g3)
        ax[0].plot(x, g4)
        ax[1].plot(nrj)
        ax[1].axhline(thresh, color='k')
        plt.show()

    nrj_filt = [x if x > thresh else np.NaN for x in nrj] 
    return nrj_filt

if __name__ == '__main__':
    file = 'data/audio/SEQ_258_AUDIO.wav'

    f, y = read(file)

    y = y.astype(np.float64)
    y /= np.linalg.norm(y)

    win = 4096
    step = 2048

    nrj = log_energie(y, f, win=win, step=step)
    nrj = np.array(nrj)

    # nrj = nrj / np.mean(nrj)

    nrj_filt = VAD(nrj, plot=True)
    # plt.plot(nrj_filt)
    # plt.show()
