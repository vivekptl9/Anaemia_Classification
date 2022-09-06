import numpy as np

path_healthy = '/Users/amateos88/Downloads/healthy.npy'
path_sick = '/Users/amateos88/Downloads/sick.npy'
data_directory = 'TO BE NAMED'

def images_anaemia_classification():
    with open(path_healthy, 'rb') as f:
        healthy = np.load(f)

    with open(path_sick, 'rb') as f:
        sick = np.load(f)
