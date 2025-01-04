import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn.externals import joblib
import joblib

import numpy as np
import streamlit as st
from PIL import Image

# Paths to directories
train_dir = r'D:\research\archive\processed_images\train'
validation_dir = r'D:\research\archive\processed_images\test'

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# CNN Model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and save CNN
cnn_model.fit(train_generator, epochs=10, validation_data=validation_generator)
cnn_model.save('cnn_model2.h5')

# Extract Features for SVM and Random Forest
train_features = cnn_model.predict(train_generator)
train_labels = train_generator.classes
test_features = cnn_model.predict(validation_generator)
test_labels = validation_generator.classes

# SVM Model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(train_features, train_labels)
joblib.dump(svm_model, 'svm_model.pkl')

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(train_features, train_labels)
joblib.dump(rf_model, 'rf_model.pkl')

# Streamlit Hybrid Model
st.title("Cataract Detection App")
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    def preprocess_image(image, target_size=(128, 128)):
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)

    processed_image = preprocess_image(image)

    cnn_prediction = cnn_model.predict(processed_image)[0][0]
    svm_features = cnn_model.predict(processed_image)
    svm_prediction = svm_model.predict_proba(svm_features)[0][1]
    rf_prediction = rf_model.predict_proba(svm_features)[0][1]

    # Combine predictions (weighted average)
    hybrid_score = (0.4 * cnn_prediction) + (0.3 * svm_prediction) + (0.3 * rf_prediction)
    result = "No Cataract" if hybrid_score > 0.5 else "Cataract Detected"

    st.write(f"Prediction: {result}")
