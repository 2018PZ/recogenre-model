import os.path
from os.path import join as pjoin
from ffmpy import FFmpeg
# spectrogram
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
# mel spectrogram
import numpy as np
import librosa.display
# neuron networks
import tensorflow as tf
from tensorflow import keras

import librosa

from src.read_data import get_train_paths


def convert_to_wav(input_path, output_path):
    if os.path.isfile(output_file):
        return 'File ' + output_file + ' already exists'
    ff = FFmpeg(inputs={input_path: None}, outputs={output_path: None})
    # ff.cmd
    ff.run()
    return 'File ' + input_path + ' was successfully converted to ' + output_path


def print_spectrogram(output_file):
    sample_rate, samples = wavfile.read(output_file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return 'Painting plot...'


def create_mel_spectrogram(output_file):
    y, sr = librosa.load(output_file)
    return librosa.feature.melspectrogram(y=y, sr=sr)


def print_mel_spectrogram(output_file):
    spec = create_mel_spectrogram(output_file)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(output_file)
    plt.show()
    # plt.tight_layout()
    return 'Painting plot...'


def create_and_save_spectrogram(output_file):
    spec = create_mel_spectrogram(output_file)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), fmax=8000)

    name = output_file[:-4]
    plt.savefig(name, bbox_inches='tight', pad_inches=-0.1, transparent=True, frameon=None, format=None)
    return 'Save sectrogram to file: ' + name + '.png'


# Preparing file
path_to_file = pjoin('../data', 'die.au')
# path_to_file = pjoin('../data', 'disco.00000.au')
# path_to_file = pjoin('../data', 'metal.00000.au')
output_file = path_to_file[:-2] + 'wav'
result = convert_to_wav(path_to_file, output_file)
print(result)

# Painting spectrogram
# print_MEL_spectrogram(output_file)

create_and_save_spectrogram(output_file)

#
# train_au_path_list = get_train_paths()
# # for file in path_list:
# file = train_au_path_list[1]
# output = file[:-2] + 'wav'
# print(output)
# convert_to_wav(file, output)


