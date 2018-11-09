
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.io.wavfile import read
from nrj import log_energie
import matplotlib.pyplot as plt
import numpy as np
# files = sorted([file for file in os.listdir(path_segmentation) if file.split('.')[-1] == 'seg'])[:2]
file = 'data/audio/SEQ_001_AUDIO.wav'

f, y = read(file)

nrj = log_energie(y, f, win=2048, step=1024)
nrj = np.array(nrj)
# plt.plot(nrj)
# plt.show()

gmm = GaussianMixture(n_components=2, covariance_type='diag')

gmm.fit(nrj.reshape(len(nrj), 1))

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

moy = gmm.means_
sig = gmm.covariances_
x = np.linspace(6, 12.5, 100)
g3 = gaussian(x, moy[0], sig[0])
g4 = gaussian(x, moy[1], sig[1])


search_grid = np.linspace(min(abs(moy)), max(abs(moy)), 100)
print(min(abs(moy)), max(abs(moy)))
g1 = gaussian(search_grid, moy[0], sig[0])
g2 = gaussian(search_grid, moy[1], sig[1])

diff = abs(g1 - g2)
thresh = x[np.argmin(diff)]

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(x, g3)
ax[0].plot(x, g4)
ax[1].plot(nrj)
ax[1].axhline(thresh, color='k')
plt.show()