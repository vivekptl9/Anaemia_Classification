import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#import Anaemia_Classification
from dataframe_creation.resizing import resize_image
from google.cloud import storage
import requests

#---------------------------------#
# Load both models
# model_cnn = load_model('cnn')
# model_r_f = load_model('random_forest')


#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")




#---------------------------------#
#Header
st.header("""
         # Detection of *Sickle* *Cell* *Disease* (Anaemia) from blood smears!
         # """)




#---------------------------------#
#Expandable About bar
expander_bar = st.expander("ABOUT THE PROJECT")
expander_bar.markdown("""
This is a LeWagon Alumni **Data** **Science** Project about
**Aneamea-detection** from blood smear images.\n
Made by
[Vivek](https://www.linkedin.com/in/vivek-patel9/),
[Ainur](https://www.linkedin.com/in/ainur-mateos-34163512b/),
[Jiazi](https://www.linkedin.com/in/wang-jiazi/)
and [Philip](https://www.linkedin.com/in/philip-steffen-71b5b823a/).
""")



#Picture of Blood smear
#image = Image.open('image.jpg')
#st.image(image, width = 200)

#st.subheader('Please upload an image to do a prediction on!')




#Loading test picture
with open('one_img.npy', 'rb') as f:
    picture = np.load(f)






#---------------------------------#
# Image Upload
st.subheader('Please upload your image for analysis below:')
uploaded_image = st.file_uploader("",type=["png","jpg","jpeg","tiff"])




#---------------------------------#
# Resize Image
if uploaded_image != None:
    img_path = "resized_image.jpg"
    resized_img = resize_image(uploaded_image)
    im = (resized_img * 255).astype(np.uint8)
    im = Image.fromarray(im)
    im.save(img_path)
#---------------------------------#
# Uploading resized img to google cloud
    client = storage.Client()
    bucket = client.bucket('anaemia')
    blob = bucket.blob('img_website_upload.jpg')
    blob.upload_from_filename(img_path)

#---------------------------------#
# Splitting Website into 2 columns (same size)

col1, col2 = st.columns((1,1))

col1.subheader('Uploaded Image')

plt.figure(figsize=(5,5), frameon=False)
if uploaded_image == None:
    plt.imshow(picture, cmap='Reds')
    col1.pyplot(plt)
    col1.text('(Shows sample images as long as nothing is uploaded)')
else:
    col1.image(im, width = 300)
# call function to rescale picture and save as uploaded_rescaled
# call function to get image with outlines and sickle cells with rectangles if there (uploaded_outlines)



col2.subheader('Computer generated Image from Upload')

plt.figure(figsize=(5,5), frameon=False)
if uploaded_image == None:
    plt.imshow(picture, cmap='Reds')
    col2.pyplot(plt)
else:
    col2.image(im, width = 300)




# loading both models
# get proba from CNN as proba_cnn and proba from random forest as proba_rf
# get proba from final model (VotingClassifier) as proba_final

# rescaled_image = rescale_image(uploaded_image)
# model_cnn = load_model('cnn')
# model_r_f = load_model('random_forest')

# proba_cnn = get_proba_cnn(model_cnn, rescaled_image) #giving the loaded model and image to get proba
# cgi , proba_r_f = get_gci_and_proba_r_f(model_r_f, rescaled_image) #cgi -> computer generated image from viveks model
# proba_final = get_final_proba(proba_cnn, proba_r_f) # Voting Classifier to get final Probability

proba_final = 99
proba_cnn = 99
proba_rf = 92

url = 'http://127.0.0.1:8000/predict'
proba_final = requests.get(url).json()['response']

st.header(f'The probability for Anaemia in the given sample is: {proba_final} %')
st.text(f'Our DeepLearning model predicts {proba_cnn}% probability and our Machine Learning model predicts {proba_rf}%')
