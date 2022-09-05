import streamlit as st
import matplotlib.pyplot as plt

#Header
st.header("""
         # Detection of *Sickle* *Cell* *Disease* (Anaemia) from blood smears!
         # """)

#Expandable About bar
expander_bar = st.expander("About")
expander_bar.markdown("""
This is a LeWagon Alumni **Data** **Science** Project about
**Aneamea-detection** from blood smear images.\n
Made by
[Vivek](https://www.linkedin.com/in/vivek-patel9/),
[Ainur](https://www.linkedin.com/in/ainur-mateos-34163512b/),
[Jiazi](https://www.linkedin.com/in/wang-jiazi/)
and [Philip](https://www.linkedin.com/in/philip-steffen-71b5b823a/).
""")

st.subheader('Please upload an image to do a prediction on!')

col1, col2 = st.columns((1,1))


uploaded_file = col1.file_uploader("Upload picture here:")

col2.text('Will show the uploaded picture here')
plt.figure(figsize=(12,10))
plt.plot([1,2], [1,2])
col2.pyplot(plt)
