''' ________ COMONLY USED LIBRARIES ________ '''
import matplotlib.pyplot as plt
from os import walk
import numpy as np
import time
''' ________ AUDIO HANDLING LIBRARY TO GET SAMPLES ________ '''
import librosa
from librosa import display
''' ________ FAST FOURIER TRANSFORM LIBRARIES ________ '''
from scipy.fft import fft, ifft

global n
global T

'''def spectrogram(samples, sample_rate, stride_ms = 10.0, window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram'''

def fft_plot(audio_samples, sampling_rate):
    #n = len(audio_samples)
    T = 1/sampling_rate
    yf = fft(audio_samples)
    xf = np.linspace(0.0, 1/(2*T) , n//2) # we create a vector to plot half od the frquencies found based on nyquist sampling teorem
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()

print("please select your current os ")
current_os
    
#mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\voice_samples_wav"
mypath = "./voice_samples_wav"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    print(f)
    
    
#samples, sampling_rate = librosa.load(mypath+"\\" + f[0], sr=None, mono=True, offset= 0.0, duration = None)
samples, sampling_rate = librosa.load(mypath+"/" + f[0], sr=None, mono=True, offset= 0.0, duration = None)
n = len(samples)
T = 1/sampling_rate

plt.figure()
librosa.display.waveplot(y = samples, sr = sampling_rate)
plt.xlabel("seconds")
plt.ylabel("amplitude")
plt.show()


fft_plot(samples, sampling_rate)

#a = spectrogram(samples, sampling_rate)