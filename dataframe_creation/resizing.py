import cv2
import matplotlib.pyplot as plt


def resize_image(img):
    img = plt.imread(img)
    img=img/255.
    img_resized = cv2.resize(img, dsize=(224,224))
    return img_resized
