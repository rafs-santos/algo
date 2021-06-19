import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

from scipy.io import wavfile
import scipy.io
import sys
import os
'''
Play Audio package
'''
import pyaudio
sys.path.insert(1, '../NeuralNet')
from torch_load_wav import playAudio


if __name__ == '__main__':

    path = "../../../../Documentos/Mestrado_UTFPR/MATLAB/Banco_de_Dados/BD/"
    names_folder = ['Saudáveis_01', 'Nódulo', 'Edema']
    Edema = []
    for file in os.listdir(path + names_folder[0]):
        if file.endswith(".wav"):
            print(os.path.join(path + "Edema", file))
            sample_rate, waveform = wavfile.read(os.path.join(path + names_folder[0], file))
            waveform = waveform.astype(np.float32, order='C') / 32768.0
            Edema.append(waveform)

    print(len(Edema))
    '''
    sample_rate, waveform = wavfile.read(path + "Edema/1253-a_n.wav")
    waveform = waveform.astype(np.float32, order='C') / 32768.0
    '''
    n_sig = 14
    print(Edema[n_sig])
    playAudio(Edema[n_sig], 50000)
    for i in range(15):
        print(len(Edema[i]))
    plt.plot(Edema[n_sig])
    plt.show()

