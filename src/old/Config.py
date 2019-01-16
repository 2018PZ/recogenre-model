# train 90, test 10
class Config:

    train_one_gender_songs = 90
    test_one_gender_songs = 10
    blues_label = 0
    classical_label = 1
    country_label = 2
    disco_label = 3
    hiphop_label = 4
    jazz_label = 5
    metal_label = 6
    pop_label = 7
    reggae_label = 8
    rock_label = 9

    @staticmethod
    def get_train_song():
        return Config.train_one_gender_songs

    @staticmethod
    def get_test_song():
        return Config.test_one_gender_songs