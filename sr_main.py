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

def transpose_matrix(mat):
    res = empty_matrix(len(mat[0]), len(mat))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res[j][i] = mat[i][j]
    return res

def empty_matrix(n,m):
    res = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(0)
        res.append(row)
    return res


def print_matrix(mat):
    res = ""
    for i in range(len(mat)):
        row = ""
        for j in range(len(mat[0])):
            row += str(mat[i][j]) + "\t"
        res += row + "\n"
    print(res)

''' ========================== OS SELECTION =========================== '''

print("please select your current os ")
print("     0. Linux / Mac ")
print("     1. Windows ")
current_os = int(input())
if current_os == 0:
    mypath = "./voice_samples_wav"    
elif current_os == 1:
    mypath = "D:\\tecdemty\\5to Semestre\\Sistems and signals\\voice_samples_wav"

''' ====================== get all test wav files ====================== '''

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    print(f)

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
    #plt.title("plot {n}".format(n = i))
    #plt.ylabel("Frequency over {n}".format(n = int(  (1/(2*T)) / len(spect)  )))
    #plt.xlabel("Time window")
    #plt.show()
    spect = numpy.array(spect)
    spect = spect.reshape(-1)
    spect = spect.tolist()
    holas.append(spect)

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
    test4 = np.array([ [0,0,0],[0,0,0],[0,1,0] ])
    test4 = test4.reshape(-1)
    test4 = test4.tolist()
    x_tests.append(test4)
    test5 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test5 = test5.reshape(-1)
    test5 = test5.tolist()
    x_tests.append(test5)
    test6 = np.array([ [0,0,0],[0,0,0],[1,0,0] ])
    test6 = test6.reshape(-1)
    test6 = test6.tolist()
    x_tests.append(test6)
    test7 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test7 = test7.reshape(-1)
    test7 = test7.tolist()
    x_tests.append(test7)
    y_tests = [[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]

    return [x_tests, y_tests]

test_data = generate_siimple_test_data()
# X_train is an array where we will store all the training spectograms
#X_train = np.array(holas)
X_train = np.array(test_data[0])
print("x train shape is {n}".format(n=str(X_train.shape)))
# Y_train is an array where we will store the answers that correspond to the training data
# results array will have the following structure ["hola", "silence"]
#Y_train = [[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0]]
#Y_train = np.array(Y_train)
Y_train = np.array(test_data[1])
print("y train is \n : {f}".format(f = Y_train))

data_inputs = X_train
data_outputs = Y_train

''' generating CNN model '''
print("data inputs[0,:]")
print(data_inputs[0,:])


HL1_neurons = 4
input_HL1_weights = numpy.random.uniform(low=-5, high=5, size=(data_inputs.shape[1], HL1_neurons))
HL2_neurons = 3
HL1_HL2_weights = numpy.random.uniform(low=-5, high=5, size=(HL1_neurons, HL2_neurons))
output_neurons = 2
HL2_output_weights = numpy.random.uniform(low=-5, high=5, size=(HL2_neurons, output_neurons))

#H1_outputs = numpy.matmul(a=data_inputs[0, :], b=input_HL1_weights)
#H1_outputs = sigmoid(H1_outputs)
#H2_outputs = numpy.matmul(a=H1_outputs, b=HL1_HL2_weights)
#H2_outputs = sigmoid(H2_outputs)
#out_otuputs = numpy.matmul(a=H2_outputs, b=HL2_output_weights)

weights = numpy_matrix_list([input_HL1_weights,
                            HL1_HL2_weights,
                            HL2_output_weights])

weights = train_network(num_iterations=100,
                        weights=weights,
                        data_inputs=data_inputs,
                        data_outputs=data_outputs,
                        learning_rate=0.01,
                        activation="sigmoid")

print("CNN trained correctly")
#predicted_label = numpy.where(out_otuputs == numpy.max(out_otuputs))[0][0]
#print("Predicted class : ", predicted_label)