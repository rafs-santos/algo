import torch
import torchaudio
from torchaudio import load
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import os

'''
Play Audio package
'''
import pyaudio
import struct

def playAudio(data, fs):
    if isinstance(data, np.ndarray):
        print("numpy array")
    else:
        data = data.numpy()

    p = pyaudio.PyAudio()

    volume = 0.5
    data =volume*data
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    frames_per_buffer=1024,
                    output=True)

    data1 = data.astype(np.float32).tobytes()
    #data1 = b''.join(struct.pack('f', samp) for samp in data) # must pack the binary data
    # play. May repeat with different volume values (if done interactively) 
    stream.write(data1)

    stream.stop_stream()
    stream.close()

    p.terminate()



def loadWav(filename, device):
    waveform, sample_rate = torchaudio.load(filename)
    #waveform, sample_rate = load(filename)
    #waveform = waveform.to(device)
    return waveform, sample_rate

if __name__ == '__main__':

    path = "../../../../Documentos/Mestrado_UTFPR/Software/Banco_de_Dados/BD/"
    #loadWav(path + "Edema/1253-a_n.wav", "cpu")
    waveform , sample_rate = loadWav(path + "Edema/1253-a_n.wav", 'cpu')
    playAudio(waveform, sample_rate)
    #wave01 = loadWav(path + "Edema/1253-a_n.wav", "cuda")
'''
    for file in os.listdir(path + "Edema"):
        if file.endswith(".wav"):
            print(os.path.join(path + "Edema", file))
'''
