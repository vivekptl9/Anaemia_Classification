import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os, sys


def resizing(new_image):
    img = plt.imread(new_image)
    #img=img/255.
    img = cv2.resize(img, dsize=(224,224))
    #f,e=os.path.splittext(path+item)
    cv2.imwrite("resized.jpg", img)

resizing()
