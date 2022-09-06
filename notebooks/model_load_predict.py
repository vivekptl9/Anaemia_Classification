from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

with open('/Users/amateos88/Desktop/Anaemia/pictures_array.npy', 'rb') as f:
    images = np.load(f)
with open('/Users/amateos88/Desktop/Anaemia/labels_array.npy', 'rb') as f:
    labels = np.load(f)





def load_model_dl(model_name: str) -> str:
    if model_name == 'DL':
        model = keras.models.load_model('/Users/amateos88/code/vivekptl9/Anaemia_Classification/Model/Inc_82_deploy')
        print('CNN model loaded..')
        return model
    elif model_name == 'Ml':
        pass
        #LOAD ML model


def predict_dl(img: np.array, model): # here should go the path of the image

    #Select the index of image to be loaded for testing
     # here the image should come as imput
    #plt.imshow(img)
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    prediction_proba = np.round(model.predict(input_img)[0][0], 2)
    return np.round(prediction_proba * 100, 2)


print(predict_dl(images[0], load_model_dl('DL')))
