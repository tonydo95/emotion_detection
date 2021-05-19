import os
import glob
from tqdm import tqdm
import numpy as np
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 32
img_height = 48
img_width = 48
AUTOTUNE = tf.data.AUTOTUNE


def parse_function(filename, label):
    """
    Read image file into 3d numpy array, convert the values in [0,1] and resize
    2D array to size [48,48]
    
    Arguments:
        filename (String) = file paths to an image
        label (int) = label corresponding to each image
    """
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, [48, 48])
    
    return image, label

def train_preprocess(image, label):
    """
    augment the image by randomly rotating, setting brightness and saturation
    
    Arguments:
    filename (String) = file paths to an image
    label (int) = label corresponding to each image
    """
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def configure_for_performance(ds):
    """
    Boost the loading images to running model

    Arguments:
        ds: tensorflow dataset
    """
    ds = ds.cache()
    ds = ds.shuffle(len(image_files))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


def get_images_and_labels(data_dir, class_names, num_files):
    image_files=[]
    labels=[]

    for class_name in class_names:
        label = class_names.index(class_name)
        
        emotion_files = glob.glob(str(data_dir.joinpath(class_name,'*')))
        constrained_files = np.random.choice(emotion_files, num_files)
        image_files.append(constrained_files)  
        labels.append([label]*num_files)

    image_files = np.asarray(image_files)
    image_files = list(image_files.flatten())
    labels = np.asarray(labels)
    labels = list(labels.flatten()) 

    return image_files, labels

def get_dataset(image_files, labels):
    ds = tf.data.Dataset.from_tensor_slices((image_files, labels))
    ds = ds.map(parse_function, num_parallel_calls=4)
    ds = ds.map(train_preprocess, num_parallel_calls=4)
    ds = configure_for_performance(ds)

    return ds