import sounddevice as sd
import peakutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift


def generateSin(F, T, fs):
    n = T*fs #numero de pontos
    x = np.linspace(0.0, T, n)  # eixo do tempo
    s = np.sin(F*x*2*np.pi)
    return (x, s)

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    #y  = np.append(signal, np.zeros(len(signal)*fs))
    N  = len(signal)
    T  = 1/fs
    xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
    yf = fft(signal)
    return(xf, fftshift(yf))

def identifyNote(note):
    #need to do
    return True




Notes = {"A": 440, "A#": 466.16, "B": 493.88, "C": 523.25, "C#": 554.37, "D": 587, "D#" : 622.25, "E": 659,  "F": 698, "F#": 740, "G": 783, "G#": 830.61}

Chords = {"A":["A", "C#", "E"], "B":["B", "D#", "F#"], "C":["C", "E", "G"], "D":["D", "F#", "A"], "E":["E", "G#", "B"], "F":["F", "A", "C"], "G":["G", "B", "D"]}


#Generate and play notes
frequency1 = 523
frequency2 = 659
frequency3 = 783
fs  = 44100
T = 1
t2   = np.linspace(-T/2,T/2,T*fs)
x1,s1 = generateSin(frequency1, T, fs)
x2,s2 = generateSin(frequency2, T, fs)
x3,s3 = generateSin(frequency3, T, fs)
finalFrequency = s1 + s2 + s3
sd.play(finalFrequency, fs)
sd.wait()


#Capture note with microphone
# duration = 3  # seconds
# t  = np.linspace(-duration/2,duration/2,duration*fs)
# myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# ymyrecording = myrecording[:,0]
# sd.wait()
# x,s = generateSin(ymyrecording, duration, fs)
# plt.plot(t,s)
# plt.xlim((0,1/100))
# plt.show()


X, Y = calcFFT(finalFrequency,fs)
plt.figure()
plt.plot(X,np.abs(Y))
plt.show()

index = peakutils.indexes(np.abs(Y), thres=0.2, min_dist=10)
print("index de picos {}" .format(index))
for freq in X[index]:
    if freq > 0 and freq < 2000:
        print(freq)
        #print(f"Frequency:{freq}")
        for k in Notes.keys():
            if (max(Notes[k],freq) % min(Notes[k], freq)) < 2:
                print(k)
            
