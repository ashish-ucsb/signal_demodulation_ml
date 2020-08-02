from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Reshape
from keras.optimizers import RMSprop

def cnn(classes, optimizer=RMSprop(lr=1e-4), objective = 'binary_crossentropy'):

    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=((28, 28))))
    
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(filters=12, kernel_size=(3, 3), padding='valid'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model
