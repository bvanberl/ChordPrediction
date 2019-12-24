from keras.layers import LSTM, Dropout, Dense, Activation, Input
from keras.models import Model

def LSTM_model(input_shape, n_vocab):
    X_input = Input(input_shape)
    X = LSTM(256, return_sequences=True)(X_input)
    X = Dropout(0.4)(X)
    X = LSTM(256)(X)
    X = Dense(256)(X)
    X = Dropout(0.4)(X)
    X = Dense(n_vocab)(X)
    Y = Activation('softmax')(X)
    model = Model(inputs=X_input, outputs=Y)
    return model