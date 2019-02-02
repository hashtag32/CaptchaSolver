import csv
import json
import h5py
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


DATASETPATH = '../../datasets/behavioral_cloning/'
LOGFILE=DATASETPATH+'big.csv'

# Reading
def read_from_csv():
    """
    Readin the data in from the csv file
    """
    image_names, steering_angles = [], []
    # Steering offset used for left and right images
    steer_offset = 0.275
    with open(LOGFILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+steer_offset, angle-steer_offset])

    return image_names, steering_angles

def shuffle_and_split(X_train, y_train):
    X_train, y_train = shuffle(X_train, y_train, random_state=14)
    return train_test_split(X_train, y_train, test_size=0.1, random_state=14)

def get_data():
    """
    Overall function for getting the X_train, X_validation, y_train, y_validation
    Returns: X_train, X_validation, y_train, y_validation
    """
    X_train, y_train=read_from_csv()
    return shuffle_and_split(X_train, y_train)



# Angle and batch processing
def getRandImgAndAngle(X_train, y_train):
    """
    Get a random imgFileName (X_train) and angle pair (y_train) 
    """
    # Get rand img + corresponding label
    selectedImg = random.randrange(len(X_train))
    selectedPosition = random.randrange(len(X_train[selectedImg]))
    angle = y_train[selectedImg][selectedPosition]
    imgFileName = DATASETPATH + str(X_train[selectedImg][selectedPosition])
    
    return imgFileName, angle

  
def validAngle(angle, low_angle_counter, batch_size):
    """
    Check whether the low_angle_counter has already enough low_angle data in the batch -> return False then. 
    """
    maxPercentageLowAngle=0.5 # per batch
    
    if(abs(angle)<0.1):
        # Angle is a low angle
        if(low_angle_counter>(batch_size*maxPercentageLowAngle)):
            # Already enough low angle data 
            return False
    return True