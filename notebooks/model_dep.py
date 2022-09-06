from asyncio import MultiLoopChildWatcher
from curses.ascii import DLE
import numpy as np
from importlib.resources import path
from matplotlib import pyplot as plt
import tensorflow as tf
from typing import Tuple
from colorama import Fore, Style
import time

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model, Input, optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping


print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()

end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")



# ARCHITECTURE: def initialize_model(X)

def initialize_model(X):

    # we define the input shape of the tensor
    input_tensor = Input(shape=(224, 224, 3))

    # instantiate the model with InceptionV3 and the desired parameters
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # we add some layers to the output of InceptionV3 model weights and layers
        # and assign it to the variable x in case we want to use it for ramdon forest
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x) #init
    x = Dense(32, activation= 'relu')(x) #init
    x = Dense(128, activation= 'relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1028, activation='relu')(x)
    x = Dense(2056, activation='relu')(x)
    x = Dense(32, activation='relu')(x) #init
    predictions = Dense(1, activation='sigmoid')(x)


    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
      layer.trainable = False

    print("✅ model initialized")
    return model, x




# COMPILE MODEL: def compile_model(model:)
def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    # $CODE_BEGIN
    optimizer = optimizers.Adam(learning_rate= learning_rate)

    model.compile(loss='binary_crossentropy',
              optimizer= 'Adam',
              metrics= 'accuracy')

    # $CODE_END

    print("\n✅ model compiled")
    return model


# we train the model
def train_model(model: Model,
                X_train,
                y_train,
                batch_size=16,
                patience=2,
                validation_split=0.2):
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    # $CODE_BEGIN
    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X_train,
                        y_train,
                        validation_split=validation_split,
                        epochs=8,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)
    # $CODE_END

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history

def evaluate_model(model: Model,
                   X_test,
                   y_test):
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        X_test,
        y_test,
        verbose=1,)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(accuracy, 2)}")

    return metrics


def accuracy_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



def load_model_dl(model_name: str) -> str:
    if model_name == 'DL':
        return keras.models.load_model('/Users/amateos88/code/vivekptl9/Anaemia_Classification/Model/Inc_82_deploy')
    elif model_name == 'Ml':
        pass
        #LOAD ML model


def predict_status_dl():
    n = 57 #Select the index of image to be loaded for testing
    img = X_test[n]
    plt.imshow(img)
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    prediction_cnn = int(np.round(model2.predict(input_img))[0][0])
    #prediction_RF = RF_model.predict(input_img_features)[0]
    #prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
    print("The predicted label for this image is: ", prediction_cnn)
    print("The actual label for this image is: ", y_test[n])
    img.shape


def predict_proba_dl():
    n = 57 #Select the index of image to be loaded for testing
    img = X_test[n]
    plt.imshow(img)
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    prediction_cnn = int(np.round(model2.predict(input_img))[0][0])
    #prediction_RF = RF_model.predict(input_img_features)[0]
    #prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
    print("The predicted label for this image is: ", prediction_cnn)
    print("The actual label for this image is: ", y_test[n])
    img.shape
