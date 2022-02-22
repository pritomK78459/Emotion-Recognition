# import packages
from pydoc import apropos
from PIL.Image import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

def build_model(input_shape:tuple):

    # this function will build a model with 5 block containing 2 convolutional
    # filters , a batch normalization layer, a maxpooling layer and a dropout layer

    model = Sequential()    # initialize a sequential model

    model.add(Rescaling(1./255))
    model.add(RandomFlip("horizontal_and_vertical"))
    model.add(RandomRotation(0.2))

    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv1_1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool1_1'))
    model.add(Dropout(0.3, name = 'drop1_1'))
    
   
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool2_1'))
    model.add(Dropout(0.3, name = 'drop2_1'))
    
    
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool3_1'))
    model.add(Dropout(0.3, name = 'drop3_1'))
    
    
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool4_1'))
    model.add(Dropout(0.3, name = 'drop4_1'))


    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), name = 'pool5_1'))


    model.add(Dropout(0.3, name = 'drop5_1'))#Flatten and output
    model.add(Flatten(name = 'flatten'))
    model.add(Dense(7, activation='softmax', name = 'output'))# create model 

    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model...')
    parser.add_argument('--train', help='path to the train directory', type=str)
    parser.add_argument('--test', help='path to the test directory', type=str)
    parser.add_argument('--batch-size', dest='batch_size',help='batch size for training', type=int)
    parser.add_argument('--epochs', help='number of epochs to train', type=int)
    parser.add_argument('--output', help='path to the directory where the model will be saved', type=str)
    parser.add_argument('--save', dest='save', help='save the model', action='store_true')
    parser.add_argument('--no-save', dest='save', help='do not save the model', action='store_false')
    parser.set_defaults(
        train='./fer_data/train', test='./fer_data/test',
        batch_size=64, epochs=50,
        output='./models/fer_tf.h5', save=True
    )

    args = parser.parse_args()

    IMG_SIZE = (48, 48) # image height and width for training

    # prepare train dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(args.train,
                                                            shuffle=True,               # shuffle the train dataset
                                                            batch_size=args.batch_size,
                                                            image_size=IMG_SIZE,
                                                             color_mode='grayscale',    # one color channel
                                                             label_mode='categorical') 

    # preparing the validation dataset
    validation_dataset = tf.keras.utils.image_dataset_from_directory(args.test,
                                                                    shuffle=True,
                                                                    batch_size=args.batch_size,
                                                                    image_size=IMG_SIZE,
                                                                    color_mode='grayscale',
                                                                    label_mode='categorical')

    optimizer =  Adam(learning_rate=0.001, decay=1e-6)  # amdam optimizer

    model = build_model(input_shape=(48,48,1))  # get the model 

    # compile all the layers and prepare the model for training
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    # train model
    history = model.fit_generator(
        train_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset
    )

    # save model
    if args.save == True:
        model.save('models/v2')
