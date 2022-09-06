import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import seaborn as sns
from sklearn import metrics

#Loading the data into a numpy array:
SIZE = int(input("Enter the value for desizing the image (Eg. 512 x 512:  "))



def data_loading():
    with open('/home/lepton/Downloads/Anaemia/Anaemia/pictures_array.npy', 'rb') as f:
        pics_all = np.load(f)
    with open('/home/lepton/Downloads/Anaemia/Anaemia/labels_array.npy', 'rb') as f:
        labels_all = np.load(f)
    #print(pics_all[0].shape)
    return pics_all,labels_all

def train_test_split(n_test_percent):
    pics_all, labels_all = data_loading()

    n_samples = len(pics_all)
    n_test = int(n_test_percent*n_samples)
    n_train = int(n_samples - n_test)

    train_set = pics_all[0:n_train]
    test_set = pics_all[n_train: n_samples]
    train_labels = labels_all[0:n_train]
    test_labels = labels_all[n_train: n_samples]

    """ Dividing the dataset in to training and test sets """
    
    X_train = train_set
    y_train = train_labels
    X_test = test_set
    y_test = test_labels
    return X_train,X_test,y_train,y_test

def feature_extension(X_train,X_test):

    activation = 'sigmoid'          #Choose the activation function as sigmoid,relu,softmax depending on your choidce and data.
    feature_ext = Sequential()      # Defining the model

    # 1st Convulation input layer
    Features = feature_ext.add(Conv2D(200,3,activation = activation,padding = "same", input_shape=(SIZE,SIZE, 3)))
    #Total layer for extraction of the features
    Features = feature_ext.add(Conv2D(150, kernel_size=(3, 3), activation=activation, padding = 'same'))
    Features = feature_ext.add(Conv2D(100, kernel_size=(3), activation=activation)) # kernel_size = 3 <==> (3, 3)
    Features = feature_ext.add(Conv2D(50, kernel_size=(3), activation=activation)) #kernel_size = 3 <==> (3, 3)
    Features = feature_ext.add(Conv2D(20, kernel_size=(3), activation=activation)) # kernel_size = 3 <==> (3, 3)

    Features = feature_ext.add(Flatten())
    X_train_RF = feature_ext.predict(X_train)
    X_test_RF = feature_ext.predict(X_test)

    return Features , X_train_RF, X_test_RF


""" Putting  """
def random_forest(n_test_percent):
    X_train,X_test,y_train,y_test = train_test_split(n_test_percent)
    Features, X_train_RF, X_test_RF = feature_extension(X_train, X_test)
    
    model = RandomForestClassifier(n_estimators=50,random_state=42)
    RF_model = model.fit(X_train_RF,y_train)
    prediction_RF = RF_model.predict(X_test_RF)
    return("Accuracy = ", metrics.accuracy_score(y_test,prediction_RF))

# def Image_prediction():
#     n=int(input(f"Give the number between 0 and {n_test}")) #Select the index of image to be loaded for testing
#     img = x_test[n]
#     plt.imshow(img)
#     input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
#     input_img_features=feature_ext.predict(input_img)
#     prediction_RF = RF_model.predict(input_img_features)[0] 
#     print("The predicted label for this image is: ", prediction_RF)
#     print("The actual label for this image is: ", test_labels[n])
#     print(img.shape)   
