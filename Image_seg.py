import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
import tensorflow as tf
import tensorflow.keras.layers
from keras.models import Sequential
import seaborn as sns
import os
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

############################################################################################################################################################
                                        #""" LOADING THE DATASETS INTO NUMPY ARRAY """
############################################################################################################################################################
SIZE = 512
def load_data():
    path = "/home/lepton/code/vivekptl9/Anaemia_Classification/DD"
    train_images = []
    for directory_path in glob.glob(path +"/Normal"):
        for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
    train_images = np.array(train_images)

    train_masks = [] 
    for directory_path in glob.glob(path+"/masks"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.tiff")):
            mask = cv2.imread(mask_path, 0)       
            mask = cv2.resize(mask, (SIZE, SIZE))
            #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            train_masks.append(mask)
            #train_labels.append(label)
        
    train_masks = np.array(train_masks)
    return train_images,train_masks

############################################################################################################################################################
                                        #""" DDIVIDING DATA INTO TRAIN AND TEST SETS """
############################################################################################################################################################

def train_test_split():
    X_train = train_images
    y_train = train_masks
    #print(y_train)
    y_train = np.expand_dims(y_train, axis=3)
    return X_train,y_train

###########################################################################################################################################################
                                       # """ EXTRACTION OF FEATURES USING CNN """
###########################################################################################################################################################

def feature_ext():

    activation = 'sigmoid'
    feature_extractor = Sequential()
    feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 3)))

    feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))

    feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))

    model_CNN = feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
   
    X = feature_extractor.predict(X_train)
    X = X.reshape(-1, X.shape[3])
    Y = y_train.reshape(-1)

    dataset = pd.DataFrame(X)
    dataset['Label'] = Y
    dataset = dataset[dataset['Label'] != 0]
    X_for_RF = dataset.drop(labels = ['Label'], axis=1)
    Y_for_RF = dataset['Label']

    return model_CNN, X_for_RF, Y_for_RF

############################################################################################################################################################
                                        #""" DEFINING RANDOM FOREST MODEL TO USE FEATURE FROM CNN """
############################################################################################################################################################
def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 50, random_state = 42)
    rf_model = model.fit(X_for_RF, Y_for_RF) 
    return rf_model

 ############################################################################################################################################################
#                                         #""" SAVING THE MODEL """
# ############################################################################################################################################################

def save_model():
    RF_model, Prediction_accuracy = random_forest()
    filename = 'CNN-RF2.sav'
    pickle.dump(RF_model, open(filename, 'wb'))
    return filename


## ############################################################################################################################################################
                                               #""" LOADING MODEL """
# ############################################################################################################################################################

def image_prediction():
    model_CNN, X_for_RF, Y_for_RF = feature_ext()
    rf_model = random_forest()

    test_img = cv2.imread('/home/lepton/Desktop/anaemia/Anaemia.png', cv2.IMREAD_COLOR)       
    test_img = cv2.resize(test_img, (SIZE, SIZE))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img = np.expand_dims(test_img, axis=0)

#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
    X_test_feature = model_CNN.predict(test_img)
    X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])

    prediction = rf_model.predict(X_test_feature)

    prediction_image = prediction.reshape(512,512)
    pred_img_show = plt.imshow(prediction_image,cmap="twilight")
    pred_img_save = plt.imsave("/home/lepton/Desktop/Anemia_rescaled.png",prediction_image, cmap="twilight")

    return pred_img_save, pred_img_show

