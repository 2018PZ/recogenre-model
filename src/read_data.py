import os
import glob


def get_train_paths():
    path = '../../../genres'
    genres = os.listdir(path)
    path_list = []

    for genre in genres:
        for p in glob.glob("../../../genres/" + genre + "/*.au"):
            if (p.__contains__('au')):
                path_list.append(p)

    return path_list


def get_wav_destinations():
    path = '../data/genres'
    path_list = []
    source = get_train_paths()

    for s in source:
        end = s.__len__()-2
        path_list.append(path + s[15:end] + 'wav')

    return path_list


def get_spec_destinations():
    path = '../data/spec'
    path_list = []
    source = get_train_paths()

    for s in source:
        end = s.__len__() - 2
        path_list.append(path + s[15:end] + "png")

    return path_list

# Test
# for p in get_train_paths():
#     print(p)