from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Input


def build_model():
    input_shape = (48, 48, 1)

    model = Sequential()

    model.add(
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1_1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1_1'))
    model.add(Dropout(0.3, name='drop1_1'))

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2_1'))
    model.add(Dropout(0.3, name='drop2_1'))

    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3_1'))
    model.add(Dropout(0.3, name='drop3_1'))

    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool4_1'))
    model.add(Dropout(0.3, name='drop4_1'))

    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool5_1'))
    model.add(Dropout(0.3, name='drop5_1'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(7, activation='softmax', name='output'))

    return model