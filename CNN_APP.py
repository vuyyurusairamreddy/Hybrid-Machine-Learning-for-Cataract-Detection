import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

cnn_model = tf.keras.models.load_model('cnn_model1.h5')

def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

st.title("Cataract Detection App")
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = cnn_model.predict(processed_image)
    result = "No Cataract" if prediction[0] > 0.5 else "Cataract Detected"
    st.write(f"Prediction: {result}")
