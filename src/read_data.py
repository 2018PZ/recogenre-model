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


# Test
# for p in get_train_paths():
#     print(p)