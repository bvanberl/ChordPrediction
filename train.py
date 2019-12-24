from sklearn.model_selection import train_test_split
from model.LSTM_model import LSTM_model
import numpy as np
import json
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

# Define some constants
DATASET_PATH = '../data/'
LOGS_PATH = '../logs/'
MODELS_PATH = '../output/'
BATCH_SIZE = 1024
EPOCHS = 3
VAL_SPLIT = 0.10

# Load data and split
X = np.load(DATASET_PATH + 'X.npy')
Y = np.load(DATASET_PATH + 'Y.npy')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

# Load token mappings
with open('./config/tokens.json') as tokens_file:
    tokens = json.load(tokens_file)
with open('./config/tokens_reverse.json') as tokens_reverse_file:
    tokens_reversed = json.load(tokens_reverse_file)

model = LSTM_model((X.shape[1], X.shape[2]), len(tokens))

tensorboard = TensorBoard(log_dir="..\\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), write_graph=False)
checkpoint = ModelCheckpoint(MODELS_PATH, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks = [tensorboard]

# Set model metrics, optimizer and loss function
metrics = metrics=["accuracy"]
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
model.summary() # Print summary of model

# Train the model
model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard], validation_split=VAL_SPLIT)

# Evaluate performance on test set
preds = model.evaluate(x=X_test, y=Y_test)
print ("Test Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
model.save(MODELS_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5")