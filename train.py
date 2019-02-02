import tensorflow as tf
import numpy as np
import csv
import cv2 
import json
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2

from preprocessing import preprocess, flipImg
from utils import get_data, validAngle, getRandImgAndAngle

# Weird workaround for cuda error, but it is working :) 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

BATCH_SIZE = 64
EPOCHS = 1

def model_nvidia():
    """
    Model according to the given Nvidia paper. Has to be defined here due to weird unicode error when model.save 
    Note. Model has to be defined here for the model.save_weights, it can't be split into a different file.
    """
    weights_regularizer=l2(0.001)

    model = Sequential()
    # Image Normalization -> lambda used
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.1))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.2))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.2))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.2))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=weights_regularizer))

    model.add(Flatten())
    model.add(Dropout(.3))
    model.add(Dense(100, activation='elu', init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='elu', init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(10, activation='elu', init='he_normal', W_regularizer=weights_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='linear', init='he_normal'))

    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    model.summary()
    return model

def get_batch(X_train, y_train):
    """
    Generate the batch for training with the data (X_train) and the corresponding grouth truth
    X_train(names):center_img.strip(), left_img.strip(), right_img.strip() 
    y_train:angle, angle+steer_offset, angle-steer_offset
    Returns: A list of image(filenames) and steeringAngles -> only for one batch
    """
    imgList = np.zeros((BATCH_SIZE, 66, 200, 3), dtype=np.float32)
    steeringAngleList = np.zeros((BATCH_SIZE,), dtype=np.float32)

    while True:
        for i in range(BATCH_SIZE):
            lowAngle_counter = 0

            # Get a valid angle (not low angle when the percentage of lowangle in the batch is already exceeded)
            while True:
                imgFileName, angle = getRandImgAndAngle(X_train,y_train)
                if not validAngle(angle, lowAngle_counter, BATCH_SIZE):
                    # Get a new data -> this one is not working
                    continue
                else:
                    # Fine. Increase and you shall pass ;)
                    lowAngle_counter += 1
                    break

            # Read image
            image = cv2.imread(imgFileName)
            # Preprocess
            image = preprocess(image)
            # Flip the image (sometimes)
            imgList[i],steeringAngleList[i]=flipImg(image,angle)

        yield imgList, steeringAngleList


if __name__=="__main__":
    # Get the training data from log file, shuffle, and split into train/validation datasets
    X_train, X_validation, y_train, y_validation = get_data()

    # Get model, print summary, and train using a generator
    model = model_nvidia()
    model.fit_generator(get_batch(X_train, y_train), samples_per_epoch=24000, nb_epoch=20, validation_data=get_batch(X_validation, y_validation), nb_val_samples=1024)#, callbacks=[early_stop])
    
    print('Training finished, model will be save to model.h5 and model.json. Call python drive.py model.json to drive in autonomous mode.')
    # Save weights to h5 and model architecture to json  
    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    print('The End.')