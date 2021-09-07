'''=======================================================================
 SPEACH RECOGNITION PROGRAM FOR 2 SECONDS PRE-RECORDED AUDIO
    DEVELOPED BY:
        - ABRIL BERENICE BAUTISTA ROMAN         ITESM CEM
        - JOSE ANGEL DEL ANGEL DOMINGUEZ        ITESM CEM
        - RAUL LOPEZ MUSITO                     ITESM CEM
        - LEONARDO JAVIER NAVA CASTELLANOS      ITESM CEM 
====================================================================='''


''' ________ COMONLY USED LIBRARIES ________ '''
import matplotlib.pyplot as plt   #ploting library
from os import walk               #library to travel trough computer paths
import numpy as np                #numeric and matrix libary
import time
import seaborn as sb              #library needed to generate heatmap
''' ________ AUDIO HANDLING LIBRARY TO GET SAMPLES ________ '''
import librosa
from librosa import display
from playsound import playsound
''' ________ FAST FOURIER TRANSFORM LIBRARIES ________ '''
from scipy.fft import fft, ifft
from scipy import signal
'''___________ CNN LIBRARY_____________'''
from cnn_model import *
from numpy_matrix_list import numpy_matrix_list
from simple_test_data import *


global n
global T
global current_os


def spectrogram(samples, sample_rate, stride_ms = 10.0, window_ms = 20.0, max_freq = None, eps = 1e-14):
    ''' spectogram function that returns a 2d list containing the normelized magnitude of each frequency 
    for each time window

    This function was retrievered from Chaudary 2020, Understanding Audio Data, Fourier Transform, FFT
    ans séctogram features for a speach recognition system at https://towardsdatascience.com/'''
    
    stride_size = int(0.001 * sample_rate * stride_ms)  # number of samples per stride i.e. 410
    window_size = int(0.001 * sample_rate * window_ms)  # number of samples per window i.e. 410

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
    return specgram

def plot_all(samples, sampling_rate, spectrogram, title):
    
    #time
    fig, ax = plt.subplots(1,3)
    xt = np.linspace(0,2,sampling_rate*2)
    ax[0].plot(xt,samples)
    ax[0].set_title("Time domain " + title)
    ax[0].set_xlabel("seconds")
    ax[0].set_ylabel("amplitude")
    
    #frequency
    yf = fft(samples)
    # we create a vector to plot half of the frquencies based
    # on nyquist sampling teorem
    xf = np.linspace(0.0, 1/(2*T) , n//2)
    ax[1].plot(xf, 2.0/n * np.abs(yf[:n//2]))
    ax[1].set_title("Frequency domain " + title)
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Magnitude")

    #spectogram
    ax[2] = sb.heatmap(spectrogram)
    ax[2].set_title(title)
    ax[2].set_ylabel("Frequency over {n}".format(n = int(  (1/(2*T)) / len(spectrogram)  )))
    ax[2].set_xlabel("Time window")
    
    return plt.show()

def plot_signal(samples, sampling_rate):
    librosa.display.waveplot(y = samples, sr = sampling_rate)
    plt.grid()
    plt.xlabel("seconds")
    plt.ylabel("amplitude")
    return plt.show()

def fft_plot(audio_samples, sampling_rate):
    yf = fft(audio_samples)
    # we create a vector to plot half of the frquencies based
    # on nyquist sampling teorem
    xf = np.linspace(0.0, 1/(2*T) , n//2) 
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()

def spectrogram_heatmap(spectrogram, title):
    sb.heatmap(spectrogram)
    plt.title(title)
    plt.ylabel("Frequency over {n}".format(n = int(  (1/(2*T)) / len(spectrogram)  )))
    plt.xlabel("Time window")
    return plt.show()

''' ========================== OS SELECTION =========================== '''

print("please select your current os ")
print("     0. Linux / Mac ")
print("     1. Windows ")
current_os = int(input())

''' ====================== WE GET ALL TRAINING "HOLA"S AUDIOS ====================== '''

print("\n============= RETRIEVING TRAINING AUDIOS ==============\n")

if current_os == 0:
    mypath = "./hola_voice_samples"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\hola_voice_samples"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    print("the files with 'holas' that will be used to train the network are: \n {f} \n".format(f = f))

# holas will be an array containing the flattened version of "hola"'s spectograms
holas = []

for i in range(len(f)):
    if current_os == 0:    
        # we extract the amplitudes (samples) of the ith audio
        samples, sampling_rate = librosa.load(mypath+"/" + f[i], sr=None, mono=True, offset= 0.0, duration = None)  
    else:
        # we extract the amplitudes (samples) of the ith audio
        samples, sampling_rate = librosa.load(mypath+"\\" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    n = len(samples)
    T = 1/sampling_rate

    # we plot signal in time and frequency domain
    # plot_signal(samples, sampling_rate)
    # fft_plot(samples, sampling_rate) 

    # we generate the spectogram of the audio with windows of 20ms
    # spectogram function will return a 2D array with the amplitude 
    # that corresponds to each amplitude in the nth window of time 
    spect = spectrogram(samples, sampling_rate, max_freq = 8000)
    
    # spectrogram_heatmap(spect, "hola {n}".format(n = i))

    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = spect.tolist()
    holas.append(spect)

''' ====================== WE GET ALL TEST "COMO"S AUDIOS ====================== '''

if current_os == 0:
    mypath = "./como_voice_samples"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\voice_samples_wav"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    print("\nthe files with 'comos' that will be used to train the network are: \n {f} \n".format(f = f))

# comos will be an array containing the flattened version of "como"'s spectograms
comos = []

for i in range(len(f)):
    if current_os == 0:    
        samples, sampling_rate = librosa.load(mypath+"/" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    else:
        samples, sampling_rate = librosa.load(mypath+"\\" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    n = len(samples)
    T = 1/sampling_rate

    # we plot signal in time and frequency domain
    # plot_signal(samples, sampling_rate)
    # fft_plot(samples, sampling_rate) 

    # we generate the spectogram of the audio with windows of 20ms
    # spectogram function will return a 2D array with the amplitude 
    # that corresponds to each amplitude in the nth window of time
    spect = spectrogram(samples, sampling_rate, max_freq = 8000)
    
    #spectrogram_heatmap(spect, "hola {n}".format(n = i))
    
    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = spect.tolist()
    comos.append(spect)

''' ====================== WE GET ALL TEST "QUE"S AUDIOS ====================== '''

if current_os == 0:
    mypath = "./que_voice_samples"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\que_voice_samples"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    print("\nthe files with 'que' that will be used to train the network are: \n {f} \n".format(f = f))

# ques will be an array containing the flattened version of "que"'s spectograms
ques = []

for i in range(len(f)):
    if current_os == 0:    
        samples, sampling_rate = librosa.load(mypath+"/" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    else:
        samples, sampling_rate = librosa.load(mypath+"\\" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    n = len(samples)
    T = 1/sampling_rate

    # we plot signal in time and frequency domain
    # plot_signal(samples, sampling_rate)
    # fft_plot(samples, sampling_rate) 

    # we generate the spectogram of the audio with windows of 20ms
    # spectogram function will return a 2D array with the amplitude 
    # that corresponds to each amplitude in the nth window of time
    spect = spectrogram(samples, sampling_rate, max_freq = 8000)
    
    #spectrogram_heatmap(spect, "hola {n}".format(n = i))
    
    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = spect.tolist()
    ques.append(spect)

''' ====================== WE TRAIN NEURAL NETWORK ====================== '''

print("\n============= TRAINING THE NEURAL NETWORK ==============\n")
Xdata = holas.copy()
Xdata += comos.copy()
Xdata += ques.copy()
X_train = np.array(Xdata)
# Y_train is an array where we will store the answers that correspond to the training data
# results will have the following structure ["hola", "como"]
Y_train = []
for i in range(len(holas)):
    Y_train.append([1,0,0])
for j in range(len(comos)):
    Y_train.append([0,1,0])
for x in range(len(ques)):
    Y_train.append([0,0,1])
'''Y_train = [[1,0],[1,0],[1,0],
            [1,0],[1,0],[1,0],
            [0,1],[0,1],[0,1],
            [0,1],[0,1],[0,1]]'''
Y_train = np.array(Y_train)
print("Y_train is :\n" + str(Y_train))
data_inputs = X_train
data_outputs = Y_train

print("\nWe will use {n} audios to test the data with {m} amplitudes \n".format(n=data_inputs.shape[0], m=data_inputs.shape[1]))

# We define the neurons and layers that our neural network will have (network architecture) 
HL1_neurons = 3200
input_HL1_weights = numpy.random.uniform(low=-1, high=1, size=(data_inputs.shape[1], HL1_neurons))
HL2_neurons = 320
HL1_HL2_weights = numpy.random.uniform(low=-1, high=1, size=(HL1_neurons, HL2_neurons))
HL3_neurons = 32
HL2_HL3_weights = numpy.random.uniform(low=-1, high=1, size=(HL2_neurons, HL3_neurons))
output_neurons = 3
HL2_output_weights = numpy.random.uniform(low=-1, high=1, size=(HL3_neurons, output_neurons))
weights = numpy_matrix_list([input_HL1_weights, HL1_HL2_weights, HL2_HL3_weights,  HL2_output_weights])

# we initialize and train our neural network with the corresponding training data 
neural_network = neuralNetwork(inital_weights = weights, trainX = data_inputs, trainY = data_outputs, learning_rate = 0.02)
neural_network.train_network(num_iterations = 200)

print("=================== CNN trained correctly ===================")

print("\n\n Let's predict training inputs, results must be : \n {f} \n\n".format(f = Y_train))
print("Prediction results for training are: \n {f} \n".format(f = neural_network.predict_outputs( X_train) ))

print("=================== AGORITHM TESTING ===================")

if current_os == 0:
    mypath = "./test_audios"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\test_audios"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)

testing = True

while testing:
    print("\nPlease select one of the following files to test the alorithm:\n")
    for i in range(len(f)):
        print(str(i) + " " + f[i])

    index = int(input())

    if current_os == 0:    
        samples, sampling_rate = librosa.load(mypath+"/" + f[index], sr=None, mono=True, offset= 0.0, duration = None)
    else:
        samples, sampling_rate = librosa.load(mypath+"\\" + f[index], sr=None, mono=True, offset= 0.0, duration = None)
    n = len(samples)
    T = 1/sampling_rate
    spect = spectrogram(samples, sampling_rate, max_freq = 8000)
    g_spect = spect.copy()
    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = np.array([spect.tolist()])

    print("\nThe file has the following audio ... \n")
    time.sleep(0.5)
    playsound(mypath +"/"+f[index])
    time.sleep(3)
    print("The file generates the following graphs")
    plot_all(samples, sampling_rate, g_spect, "audio {n}".format(n = index+1))
    print("Prediction structure is: \n ['hola','como', 'que'] \n")
    print("\nPrediction results are: \n {f} \n".format(f = neural_network.predict_outputs(spect) ))
    print("Do you want to try another audio? \n0 no \n1 yes")
    testing = bool(int(input()))