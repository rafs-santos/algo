import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pywt
import pywt.data

import pyaudio
import struct


def playAudio(data, fs):
    p = pyaudio.PyAudio()

    volume = 0.15
    data =volume*data
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    frames_per_buffer=1024,
                    output=True)

    data1 = b''.join(struct.pack('f', samp) for samp in data) # must pack the binary data
    # play. May repeat with different volume values (if done interactively) 
    stream.write(data1)

    stream.stop_stream()
    stream.close()

    p.terminate()

if __name__ == '__main__':
    path = "../BD/BD.mat"
    test = scipy.io.loadmat(path)
    fs = test['fs'][0][0]
    edema = test['edema']
    #print(edema.shape)
    audio1 = np.array(edema[::,7]).astype(np.float32)
    playAudio(audio1,50000)

    t = np.linspace(0,0.5,25000)
    plt.plot(t,edema[::,7])
    plt.show()

    '''
    # load image
    original = pywt.data.camera()

    # wavelet transform of image, and plot approximation and details
    titles = ['approximation', ' horizontal detail',
              'vertical detail', 'diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    ll, (lh, hl, hh) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([ll, lh, hl, hh]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
    '''
