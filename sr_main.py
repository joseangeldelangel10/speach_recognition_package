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
import numpy as np                
import time
import seaborn as sb              #library needed to generate heatmap
''' ________ AUDIO HANDLING LIBRARY TO GET SAMPLES ________ '''
import librosa
from librosa import display
''' ________ FAST FOURIER TRANSFORM LIBRARIES ________ '''
from scipy.fft import fft, ifft
from scipy import signal
'''___________ CNN LIBRARY_____________'''
from cnn_model import *
from numpy_matrix_list import numpy_matrix_list


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
    print("window_size is {f}".format(f=window_size))

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

def fft_plot(audio_samples, sampling_rate):
    yf = fft(audio_samples)
    xf = np.linspace(0.0, 1/(2*T) , n//2) # we create a vector to plot half od the frquencies found based on nyquist sampling teorem
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()

''' ========================== OS SELECTION =========================== '''

print("please select your current os ")
print("     0. Linux / Mac ")
print("     1. Windows ")
current_os = int(input())

if current_os == 0:
    mypath = "./hola_voice_samples"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\voice_samples_wav"

''' ====================== get all test wav files ====================== '''

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    #print(f)

''' ================== GET SAMPLES FROM 1ST FILE ====================== '''    

holas = []

for i in range(len(f)):
    if current_os == 0:    
        samples, sampling_rate = librosa.load(mypath+"/" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    else:
        samples, sampling_rate = librosa.load(mypath+"\\" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    n = len(samples)
    T = 1/sampling_rate

    ''' ======================= MAKE PLOTS S & T======================== 
    plt.figure()
    librosa.display.waveplot(y = samples, sr = sampling_rate)
    plt.xlabel("seconds")
    plt.ylabel("amplitude")
    plt.show()
    fft_plot(samples, sampling_rate) '''

    spect = spectrogram(samples, sampling_rate, max_freq = 8000)
    fig, ax = plt.subplots()
    #sb.heatmap(spect)
    #plt.title("hola {n}".format(n = i))
    #plt.ylabel("Frequency over {n}".format(n = int(  (1/(2*T)) / len(spect)  )))
    #plt.xlabel("Time window")
    #plt.show()
    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = spect.tolist()
    holas.append(spect)




if current_os == 0:
    mypath = "./como_voice_samples"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\voice_samples_wav"

''' ====================== get all test wav files ====================== '''

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    #print(f)

''' ================== GET SAMPLES FROM 1ST FILE ====================== '''    

comos = []

for i in range(len(f)):
    if current_os == 0:    
        samples, sampling_rate = librosa.load(mypath+"/" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    else:
        samples, sampling_rate = librosa.load(mypath+"\\" + f[i], sr=None, mono=True, offset= 0.0, duration = None)
    n = len(samples)
    T = 1/sampling_rate

    ''' ======================= MAKE PLOTS S & T======================== 
    plt.figure()
    librosa.display.waveplot(y = samples, sr = sampling_rate)
    plt.xlabel("seconds")
    plt.ylabel("amplitude")
    plt.show()
    fft_plot(samples, sampling_rate) '''

    spect = spectrogram(samples, sampling_rate, max_freq = 8000)
    fig, ax = plt.subplots()
    #sb.heatmap(spect)
    #plt.title("como {n}".format(n = i))
    #plt.ylabel("Frequency over {n}".format(n = int(  (1/(2*T)) / len(spect)  )))
    #plt.xlabel("Time window")
    #plt.show()
    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = spect.tolist()
    comos.append(spect)


def generate_siimple_test_data():
    '''function that genrates simple data 3x3 bits and uses CNN to check if dot is centered or outside'''
    x_tests = []
    test1 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test1 = test1.reshape(-1)
    test1 = test1.tolist()
    x_tests.append(test1)
    test2 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test2 = test2.reshape(-1)
    test2 = test2.tolist()
    x_tests.append(test2)
    test3 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test3 = test3.reshape(-1)
    test3 = test3.tolist()
    x_tests.append(test3)
    test4 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test4 = test4.reshape(-1)
    test4 = test4.tolist()
    x_tests.append(test4)
    test5 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test5 = test5.reshape(-1)
    test5 = test5.tolist()
    x_tests.append(test5)
    test6 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test6 = test6.reshape(-1)
    test6 = test6.tolist()
    x_tests.append(test6)
    test7 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test7 = test7.reshape(-1)
    test7 = test7.tolist()
    x_tests.append(test7)
    y_tests = [[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[1,0]]

    return [x_tests, y_tests]

#test_data = generate_siimple_test_data()
# X_train is an array where we will store all the training spectograms
Xdata = holas
Xdata += comos
X_train = np.array(Xdata)
#X_train = np.array(test_data[0])
# Y_train is an array where we will store the answers that correspond to the training data
# results array will have the following structure ["hola", "como"]
Y_train = [[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
Y_train = np.array(Y_train)
#Y_train = np.array(test_data[1])

data_inputs = X_train
data_outputs = Y_train

''' generating CNN model '''
#print("data inputs[0,:]")
#print(data_inputs[0,:])

HL1_neurons = X_train.shape[0]//2
input_HL1_weights = numpy.random.uniform(low=-1.2, high=1.2, size=(data_inputs.shape[1], HL1_neurons))
HL2_neurons = 1000
HL1_HL2_weights = numpy.random.uniform(low=-1.2, high=1.2, size=(HL1_neurons, HL2_neurons))
output_neurons = 2
HL2_output_weights = numpy.random.uniform(low=-1.2, high=1.2, size=(HL2_neurons, output_neurons))

weights = numpy_matrix_list([input_HL1_weights, HL1_HL2_weights, HL2_output_weights])

#X_train, weights = neuralNetwork.add_bias_terms(X_train, weights)

neural_network = neuralNetwork(inital_weights = weights, trainX = data_inputs, trainY = data_outputs, learning_rate = 0.01)
neural_network.train_network(num_iterations = 100)

print("=================== CNN trained correctly ===================")

print("\n\n Let's predict some inputs, results must be : \n {f} \n\n".format(f = Y_train))
print("Prediction results are: \n {f} \n".format(f = neural_network.predict_outputs( X_train) ))

#predicted_label = numpy.where(out_otuputs == numpy.max(out_otuputs))[0][0]
#print("Predicted class : ", predicted_label)