# compatibility with python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import librosa as dps
import tensorflow.contrib as tfc
import os as os
import matplotlib.pyplot as plt
import librosa.display as ldsp

'''
gabkax = 0
gabkay = 1
gabkaz = 2
papierx = 3
papiery = 4
papierz = 5
metalx = 6
metaly = 7
metalz = 8
plastikx = 9
plastiky = 10
plastikz = 11
gumkax = 12
gumkay = 13
gumkaz = 14

'''

filenames = ('Materials/plastik/plastik1.txt')



spx, spy, spz = np.genfromtxt(filenames, delimiter=' ', unpack=True, dtype=None)



nspx = np.resize(spx, (396,))

nspy = np.resize(spy, (396,))

nspz = np.resize(spz, (396,))


x_vect = []
y_vect = []
z_vect = []
label_vect = []


for x in np.arange(42):
    randomspx = np.random.normal(0, np.random.randint(0.5, 25), 396)
    new_vector = nspx + randomspx
    x_vect.append(new_vector)


for x in np.arange(42):
    randomspx = np.random.normal(0, np.random.randint(0.5, 7), 396)
    new_vector = nspy + randomspx
    y_vect.append(new_vector)


for x in np.arange(42):
    randomspx = np.random.normal(0, np.random.randint(0.5, 25), 396)
    new_vector = nspz + randomspx
    z_vect.append(new_vector)
print("x")
'''
for x in np.arange(len(features_np)):
    sig, fs = dps.core.load(features_np[x])
    S = dps.feature.melspectrogram(y=sig, sr=fs)
    spectr_vect.append(S)
'''


for x in np.arange(len(x_vect)):
    sig = x_vect[x]
    fs = 22500
    print("krok:", x)
    S = dps.feature.melspectrogram(y=sig, sr=fs)
    S = ldsp.specshow(dps.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    print(type(S))
    zmienna = 'pliki/evaluation/spectrograms_eval/output' + str(x+252) + '.png'
    S.plot()
    fig = plt.gcf()
    fig.savefig(zmienna)

for x in np.arange(len(y_vect)):
    sig = y_vect[x]
    fs = 22500
    print("krok:", x)
    S = dps.feature.melspectrogram(y=sig, sr=fs)
    S = ldsp.specshow(dps.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    print(type(S))
    zmienna = 'pliki/evaluation/spectrograms_eval/output' + str(x+294) + '.png'
    S.plot()
    fig = plt.gcf()
    fig.savefig(zmienna)

for x in np.arange(len(z_vect)):
    sig = z_vect[x]
    fs = 22500
    print("krok:", x)
    S = dps.feature.melspectrogram(y=sig, sr=fs)
    S = ldsp.specshow(dps.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    print(type(S))
    zmienna = 'pliki/evaluation/spectrograms_eval/output' + str(x+336) + '.png'
    S.plot()
    fig = plt.gcf()
    fig.savefig(zmienna)

