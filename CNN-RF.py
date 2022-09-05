import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import seaborn as sns

#Loadind the data into a numpy array:

def Data_loading():
    with open('/home/lepton/Downloads/Anaemia/Anaemia/pictures_array.npy', 'rb') as f:
        pics_all = np.load(f)
    with open('/home/lepton/Downloads/Anaemia/Anaemia/labels_array.npy', 'rb') as f:
        labels_all = np.load(f)
    print(pics_all[0].shape)
    return pics_all,labels_all