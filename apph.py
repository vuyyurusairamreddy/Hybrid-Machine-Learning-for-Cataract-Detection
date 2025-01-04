import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

# Load trained models
cnn_model = tf.keras.models.load_model('cnn_model2.h5')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Function to preprocess the image
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Hybrid prediction function
def hybrid_prediction(image):
    # Get CNN prediction
    cnn_prediction = cnn_model.predict(image)[0][0]
    
    # Get SVM and Random Forest predictions
    features = cnn_model.predict(image)  # CNN feature extraction
    svm_prediction = svm_model.predict_proba(features)[0][1]  # Probability of cataract
    rf_prediction = rf_model.predict_proba(features)[0][1]  # Probability of cataract
    
    # Weighted average for hybrid model
    hybrid_score = (0.4 * cnn_prediction) + (0.3 * svm_prediction) + (0.3 * rf_prediction)
    result = "No Cataract" if hybrid_score < 0.5 else "Cataract Detected"
    
    return result, hybrid_score, cnn_prediction, svm_prediction, rf_prediction

# Streamlit UI
st.title("Hybrid Cataract Detection App")
st.write("Upload an eye image to detect if cataract is present.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get predictions
    result, hybrid_score, cnn_score, svm_score, rf_score = hybrid_prediction(processed_image)
    
    # Display results
    st.subheader("Prediction Results:")
    st.write(f"**Hybrid Model Result:** {result}")
    st.write(f"**Hybrid Score:** {hybrid_score:.2f}")
    st.write(f"**CNN Confidence:** {cnn_score:.2f}")
    st.write(f"**SVM Confidence:** {svm_score:.2f}")
    st.write(f"**Random Forest Confidence:** {rf_score:.2f}")
