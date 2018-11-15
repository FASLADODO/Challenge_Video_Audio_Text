import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy import signal

def log_energie(Y, sr, win=512, step=256, plot=False):
    """
    Calcul de la log énergie d'un signal sur des fenêtres glissantes

    Arguments:
        Y {[numpy array]} -- signal dont on veut l'énergie
        win {int} -- Taille des fenêtres glissantes en nombre de points (defaut: {512})
        step {int} -- Taille du pas des fenêtres glissantes en nombre de points (defaut: {256})
        plot {bool} -- Affiche ou non le résultat obtenu (defaut: {False})

    Returns:
        [list] -- Log énergie du signal
    """

    Y = Y.astype(np.float64)
    N = len(Y)
    nrj = []
    for i in range(0, N, step):
        if i+win < N:
            nrj.append(np.log10(np.sum(np.power(Y[i:i+win], 2))))

    time = np.cumsum([step/sr]*len(nrj))

    if plot:
        x = np.cumsum([step/sr] * len(nrj))
        plt.figure(1, figsize=(12, 8))
        plt.plot(x, nrj)
        plt.title("Log(Enegie)")
        plt.show()
    return nrj, time

if __name__ == '__main__':
    f, Y = read('data/audio/SEQ_001_AUDIO.wav')
    nrj = log_energie(Y, f, 4096, 2048, plot=True)

    b, a = signal.butter(3, 0.1)
    nrj = signal.filtfilt(b, a, nrj)

    plt.plot(nrj)
    plt.show()