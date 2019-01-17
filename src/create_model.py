from src.common import GENRES
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation, TimeDistributed, Convolution1D, \
    MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
import pickle
from optparse import OptionParser
import os

SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 10


def create_data_sets(data):
    x = data['x']
    y = data['y']
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3, random_state=SEED)
    return x_train, x_val, y_train, y_val


def build_model(x_train):
    print('Building model...')

    n_features = x_train.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    layer = model_input
    for i in range(N_LAYERS):
        # second convolutional layer names are used by extract_filters.py
        layer = Convolution1D(
            filters=CONV_FILTER_COUNT,
            kernel_size=FILTER_LENGTH,
            name='convolution_' + str(i + 1)
        )(layer)
        # layer = BatchNormalization(momentum=0.9)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
        # layer = Dropout(0.5)(layer)

    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    time_distributed_merge_layer = Lambda(
        function=lambda x: K.mean(x, axis=1),
        output_shape=lambda shape: (shape[0],) + shape[2:],
        name='output_merged'
    )
    layer = time_distributed_merge_layer(layer)
    layer = Activation('softmax', name='output_realtime')(layer)  # softplus
    model_output = layer
    model = Model(model_input, model_output)
    opt = Adam(lr=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    print(model.summary())
    return model


def train_model(x_train, y_train, x_val, y_val, model_path, model):
    print('Training...')
    model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, validation_data=(x_val, y_val), verbose=1,
        callbacks=[
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_acc', verbose=1),
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_delta=0.01, verbose=1)
        ]
    )
    return model


def evaluate_model(x_val, y_val, model):
    info = model.evaluate(x_val, y_val, verbose=0)
    print("SCORE: ", info[1])
    # model.predict(x_train[0])
    return info


def predict_model(x_test, model):
    result = model.predict(x_test, batch_size=BATCH_SIZE)
    # print(result)
    for i in result:
        print(i)
    return result


def load_trained_model(x_train, weights_path):
    model = build_model(x_train)
    model.load_weights(weights_path)
    return model


def create_model(data, model_path):
    (x_train, x_val, y_train, y_val) = create_data_sets(data)

    model = build_model(x_train)

    model = train_model(x_train, y_train, x_val, y_val, model_path, model)

    evaluate_model(x_val, y_val, model)

    return model


def create_model_from_file(data, model_path):
    (x_train, x_val, y_train, y_val) = create_data_sets(data)

    model = load_trained_model(x_train, model_path)

    evaluate_model(x_val, y_val, model)

    predict_model(x_val, model)

    return model


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--data_path', dest='data_path', default=os.path.join('../', 'data/musicData.pkl'),
            help='path to the data pickle', metavar='DATA_PATH')
    parser.add_option('-m', '--model_path', dest='model_path', default=os.path.join('../', 'models/model10e.h5'),
            help='path to the output model HDF5 file', metavar='MODEL_PATH')
    options, args = parser.parse_args()

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    # create_model(data, options.model_path)
    create_model_from_file(data, options.model_path)
