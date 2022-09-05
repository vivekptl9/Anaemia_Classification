import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import Anaemia_Classification

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
with open('notebooks/pictures_array.npy', 'rb') as f:
    pictures = np.load(f)

#---------------------------------#
# Image Upload
st.subheader('Please upload your image for analysis below:')
uploaded_image = st.file_uploader("")

#---------------------------------#
# Splitting Website into 2 columns (same size)

col1, col2 = st.columns((1,1))

col1.subheader('Uploaded Image')

plt.figure(figsize=(5,5), frameon=False)
if uploaded_image == None:
    plt.imshow(pictures[0], cmap='Reds')
    col1.pyplot(plt)
else:
    plt.imshow(uploaded_image, cmap='Reds')
    col1.pyplot(plt)

# call function to rescale picture and save as uploaded_rescaled
# call function to get image with outlines and sickle cells with rectangles if there (uploaded_outlines)

col2.subheader('Computer generated Image from Upload')
st.text('(Shows sample images as long as nothing is uploaded)')
plt.figure(figsize=(5,5), frameon=False)
if uploaded_image == None:
    plt.imshow(pictures[0], cmap='Reds')
    col2.pyplot(plt)
else:
    plt.imshow(uploaded_image, cmap='Reds')
    col2.pyplot(plt)




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
proba_cnn = 95
proba_rf = 92

st.header(f'The probabilty for Anaemia in the given sample is: {proba_final} %')
st.text(f'Our DeepLearning model predicts {proba_cnn}% probability and our Machine Learning model predicts {proba_rf}%')
