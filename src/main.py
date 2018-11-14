import os.path
from os.path import join as pjoin
from ffmpy import FFmpeg


def convert_to_wav(input_path, output_path):
    if os.path.isfile(output_file):
        return 'File ' + output_file + ' already exists'
    ff = FFmpeg(inputs={input_path: None}, outputs={output_path: None})
    ff.cmd
    ff.run()
    return 'File ' + input_path + ' was successfuly converted to ' + output_path


path_to_file = pjoin('data', 'die.au')
output_file = path_to_file[:-2] + 'wav'
result = convert_to_wav(path_to_file, output_file)
print(result)