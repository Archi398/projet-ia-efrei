import streamlit as st
import requests
from PIL import Image

st.title("Image Classification App")

st.write("Upload an image of a dog or a cat to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Predict'):
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict", files=files)
        prediction = response.json()
        st.write(prediction)
        
if st.button('Download Model'):
    with open("model/model.h5", "rb") as file:
        st.download_button(label="Download Model", data=file, file_name="model.h5")
