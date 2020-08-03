import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from scipy.io import loadmat

from visualization_block import visualization_block 

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Reshape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils import Sequence

# Ignore Warnings
warnings.simplefilter("ignore", UserWarning)

# Custom ImageGenerator
class ImageGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = [visualization_block(r) for r in batch_x] 
        y = [r for r in batch_y]
        return np.array(x), np.array(y)

# CNN Model
def cnn(classes, optimizer=RMSprop(lr=1e-4), objective = 'binary_crossentropy'):
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=((28, 28))))
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    model.add(Conv2D(filters=12, kernel_size=(3, 3), padding='valid'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

# Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

# Prepare Data
x = loadmat('data/ook_10p_0cm.mat')['data_10p_0cm'].T
y = loadmat('data/ook_10p_label.mat')['org_label'][0]
classes = len(unique_labels(y))
print("Original Data: {}".format(x.shape))
print("Original Labels: {}".format(y.shape))

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, to_categorical(y, num_classes=None), test_size=0.30, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.30, random_state=42)
print("Train: {}, Test: {}, Validation: {}".format(y_train.shape, y_test.shape, y_validation.shape))

# Parameters
nb_epoch = 1
batch_size = 50
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, mode='auto')
history = LossHistory()

# Generators
train_gen = ImageGenerator(x_train, y_train, batch_size)
test_gen = ImageGenerator(x_test, y_test, batch_size)

# Train
model = cnn(classes)
model.fit(train_gen, epochs=nb_epoch, verbose=1, shuffle=True, callbacks=[history, early_stopping])
print(model.summary())

# Predict
predictions = model.predict_generator(test_gen)
print(predictions)