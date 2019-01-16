import os.path
from ffmpy import FFmpeg
# spectrogram
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
# mel spectrogram
import numpy as np
import librosa.display
import librosa

from src.common import create_mel_spectrogram
from src.read_data import get_train_paths
from src.read_data import get_wav_destinations
from src.read_data import get_spec_destinations


def convert_to_wav(input_path, output_path):
    if os.path.isfile(output_path):
        return 'File ' + output_path + ' already exists'
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


def print_mel_spectrogram(output_file):
    spec = create_mel_spectrogram(output_file)

    plt.figure(figsize=(10, 4))
    # librosa.display.specshow(spec, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title(output_file)
    plt.show()
    # plt.tight_layout()
    return 'Painting plot...'


def create_and_save_spectrogram(input_file, output_file):
    spec = create_mel_spectrogram(input_file)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), fmax=8000)

    plt.savefig(output_file, bbox_inches='tight', pad_inches=-0.1, transparent=True, frameon=None, format=None)
    return 'Save sectrogram to file: ' + output_file + '.png'


# Preparing file
output_file = '../../data/genres/blues/blues.00000.au'
# Painting spectrogram
print_mel_spectrogram(output_file)
# create_and_save_spectrogram(output_file)


# train_au_path_list = get_train_paths()
# train_wav_path_list = get_wav_destinations()
# nuber_of_files = train_wav_path_list.__len__()
#
# for x in range(0, nuber_of_files):
#     convert_to_wav(train_au_path_list[x], train_wav_path_list[x])
#
# spectrograms = get_spec_destinations()
#
# print(spectrograms)
#
# for x in range(0, nuber_of_files):
#     create_and_save_spectrogram(train_wav_path_list[x], spectrograms[x])


# for file in path_list:
# file = train_au_path_list[1]
# output = file[:-2] + 'wav'
# print(output)
# convert_to_wav(file, output)


