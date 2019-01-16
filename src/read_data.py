import os
import glob

from src.old import Config


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
        end = s.__len__() - 2
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

# DATA SPECTROGRAM'S SETS PREPARATION
train_blues = sorted(glob.glob("../data/spec/blues/*.png"))
train_classical = sorted(glob.glob("../data/spec/classical/*.png"))
train_country = sorted(glob.glob("../data/spec/country/*.png"))
train_disco = sorted(glob.glob("../data/spec/disco/*.png"))
train_hiphop = sorted(glob.glob("../data/spec/hiphop/*.png"))
train_jazz = sorted(glob.glob("../data/spec/jazz/*.png"))
train_metal = sorted(glob.glob("../data/spec/metal/*.png"))
train_pop = sorted(glob.glob("../data/spec/pop/*.png"))
train_reggae = sorted(glob.glob("../data/spec/reggae/*.png"))
train_rock = sorted(glob.glob("../data/spec/rock/*.png"))

train_labels = [0]
train_set = []
test_labels = []
test_set = []
train_one_gender_songs = Config.Config.get_train_song()

for x in range(1, len(train_blues)):
    label = Config.Config.blues_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_blues.pop())
for x in range(0, len(train_classical)):
    label = Config.Config.classical_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_classical.pop())
for x in range(0, len(train_country)):
    label = Config.Config.country_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_country.pop())
for x in range(0, len(train_disco)):
    label = Config.Config.disco_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_disco.pop())
for x in range(0, len(train_hiphop)):
    label = Config.Config.hiphop_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_hiphop.pop())
for x in range(0, len(train_jazz)):
    label = Config.Config.jazz_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_jazz.pop())
for x in range(0, len(train_metal)):
    label = Config.Config.metal_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_metal.pop())
for x in range(0, len(train_pop)):
    label = Config.Config.pop_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_pop.pop())
for x in range(0, len(train_reggae)):
    label = Config.Config.reggae_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_reggae.pop())
for x in range(0, len(train_rock)):
    label = Config.Config.rock_label
    if x < train_one_gender_songs:
        train_labels.append(label)
    else:
        test_labels.append(label)
        test_set.append(train_rock.pop())

train_set = train_blues + train_classical + train_country + train_disco + train_hiphop + train_jazz + train_metal \
            + train_pop + train_reggae + train_rock


def get_train_set():
    return train_set


def get_train_labels():
    return train_labels


def get_test_set():
    return test_set


def get_test_labels():
    return test_labels
