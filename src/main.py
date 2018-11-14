import os.path
from os.path import join as pjoin
from ffmpy import FFmpeg
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def convert_to_wav(input_path, output_path):
    if os.path.isfile(output_file):
        return 'File ' + output_file + ' already exists'
    ff = FFmpeg(inputs={input_path: None}, outputs={output_path: None})
    ff.cmd
    ff.run()
    return 'File ' + input_path + ' was successfully converted to ' + output_path


def print_spectogram(output_file):
    sample_rate, samples = wavfile.read(output_file)
    fraquencies, times, spectogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, fraquencies, spectogram)
    plt.imshow(spectogram)
    plt.ylabel('Fraquency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return 'Painting plot...'


# Preparing file
path_to_file = pjoin('data', 'die.au')
output_file = path_to_file[:-2] + 'wav'
result = convert_to_wav(path_to_file, output_file)
print(result)

# Painting spectogram
print_spectogram(output_file)