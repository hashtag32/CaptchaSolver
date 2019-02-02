import numpy as np
import random
import cv2 

def grayscale(image):
    """
    Grayscaling the image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize(image):
    """
    Resizing the image to 200,66
    """
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

def crop_image(image):
    """
    Cropping the image
    Cut off 43 pixels from the top and -24 from the bottom
    """
    return image[43:-24,:]

def brightness(image):
    """
    Returns an image with a random degree of brightness.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def preprocess(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    
    image= grayscale(image)
    image = brightness(image)
    image = crop_image(image)
    image = resize(image)
    return np.array(image, dtype=np.float32)

def flipImg(image, angle):
    """
    Returns an image after flipping it in 50% of the cases
    :param image: Image represented as a numpy array.
    """
    if random.randrange(2) == 1:
        image = cv2.flip(image, 1)
        angle = -angle
    return image,angle
