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
import pickle

############################################################################################################################################################
                                        #""" LOADING THE DATASETS INTO NUMPY ARRAY """
############################################################################################################################################################

def data_loading():
    with open('/home/lepton/Downloads/Anaemia/Anaemia/pictures_array.npy', 'rb') as f:
        pics_all = np.load(f)
    with open('/home/lepton/Downloads/Anaemia/Anaemia/labels_array.npy', 'rb') as f:
        labels_all = np.load(f)
    return pics_all,labels_all

############################################################################################################################################################
                                        #""" DDIVIDING DATA INTO TRAIN AND TEST SETS """
############################################################################################################################################################

def train_test_split():
    pics_all, labels_all = data_loading()

    n_samples = len(pics_all)-300
    n_test = int(0.2*n_samples)
    n_train = int(n_samples - n_test)

    X_train = pics_all[0:n_train]
    X_test = pics_all[n_train: n_samples]
    y_train = labels_all[0:n_train]
    y_test = labels_all[n_train: n_samples]
    print(X_train,X_test)
    return X_train,X_test,y_train,y_test

###########################################################################################################################################################
                                       # """ EXTRACTION OF FEATURES USING CNN """
###########################################################################################################################################################
    
def feature_extension():
    X_train,X_test,y_train,y_test = train_test_split()
    activation = 'sigmoid'          #Choose the activation function as sigmoid,relu,softmax depending on your choidce and data.
    feature_ext = Sequential()      # Defining the model

    # 1st Convulation input layer
    Features = feature_ext.add(Conv2D(200,3,activation = activation,padding = "same", input_shape=(224,224, 3)))
    #Total layer for extraction of the features
    Features = feature_ext.add(Conv2D(150, kernel_size=(3, 3), activation=activation, padding = 'same'))
    Features = feature_ext.add(Conv2D(100, kernel_size=(3), activation=activation)) # kernel_size = 3 <==> (3, 3)
    Features = feature_ext.add(Conv2D(50, kernel_size=(3), activation=activation)) #kernel_size = 3 <==> (3, 3)
    Features = feature_ext.add(Conv2D(20, kernel_size=(3), activation=activation)) # kernel_size = 3 <==> (3, 3)

    Features = feature_ext.add(Flatten())
    X_train_RF = feature_ext.predict(X_train)
    X_test_RF = feature_ext.predict(X_test)

    return Features , X_train_RF, X_test_RF


############################################################################################################################################################
                                        #""" DEFINING RANDOM FOREST MODEL TO USE FEATURE FROM CNN """
############################################################################################################################################################
def random_forest():
    X_train,X_test,y_train,y_test = train_test_split()
    Features, X_train_RF, X_test_RF = feature_extension()
    #Training the feature extracted training set with Random forest
    model = RandomForestClassifier(n_estimators=50,random_state=42)
    RF_model = model.fit(X_train_RF,y_train)
    # Testing the feature extracted training set with Random forest
    prediction_RF = model.predict(X_test_RF)
    Prediction_accuracy = metrics.accuracy_score(y_test,prediction_RF)
    return RF_model,Prediction_accuracy


# ############################################################################################################################################################
#                                         #""" SAVING THE MODEL """
# ############################################################################################################################################################

def save_model():
    RF_model, Prediction_accuracy = random_forest()
    filename = 'CNN-RF2.sav'
    pickle.dump(RF_model, open(filename, 'wb'))
    return filename

# ############################################################################################################################################################
#                                         #""" LOADING MODEL """
# ############################################################################################################################################################

def Image_prediction():
    filename = save_model()
    RF_model, Prediction_accuracy = random_forest()
    X_train,X_test,y_train,y_test = train_test_split()
    model =  pickle.load(open(filename, "rb"))
    return Prediction_accuracy  

# Image_prediction()
    
    
#     # n=int(input(f"Give the number between 0 and {len(X_test)}")) #Select the index of image to be loaded for testing
#     # img = X_test[n]
#     # plt.imshow(img)
#     # input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
#     # input_img_features=feature_ext.predict(input_img)
#     # prediction_RF = RF_model.predict(input_img_features)[0] 
#     # print("The predicted label for this image is: ", prediction_RF)
#     # print("The actual label for this image is: ", test_labels[n])
#     # print(img.shape)   
