import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
import gdown
import os

# Download the model file from Google Drive
def download_model():
    model_url = "https://drive.google.com/file/d/1lDBpPogpcNfdPWl3iw2GyX2JHPuWMppj/view?usp=drive_link"
    output_path = "brain_tumor_model.h5"
    
    if not os.path.exists(output_path):
        st.write("Downloading model from Google Drive...")
        gdown.download(model_url, output_path, quiet=False)
        st.write("Model downloaded successfully!")
    
    return tf.keras.models.load_model(output_path)

# Load the model
model = download_model()

# Define the class names based on your dataset
class_names = ['no_tumor', 'meningioma', 'glioma', 'pituitary_tumor']

# Tumor advice dictionary
tumor_advice = {
    'no tumor': [
        {"point": "Stay Vigilant", "advice": "Even if no tumor is detected, regular check-ups and healthy lifestyle choices are crucial for brain health."}
    ],
    'meningioma': [
        {"point": "Monitor Symptoms", "advice": "Regular MRI scans and neurological assessments are recommended to track the tumor's size and potential effects."},
        {"point": "Treatment Options", "advice": "Discuss surgical and radiation therapy options with your healthcare provider if symptoms worsen."}
    ],
    'glioma': [
        {"point": "Seek Specialist Care", "advice": "Consult a neuro-oncologist for a comprehensive treatment plan, including surgery, radiation, or chemotherapy."},
        {"point": "Symptom Management", "advice": "Medications may help manage symptoms like seizures, headaches, and cognitive difficulties."}
    ],
    'pituitary_tumor': [
        {"point": "Hormone Testing", "advice": "Regular blood tests to monitor hormone levels and assess pituitary gland function."},
        {"point": "Vision Monitoring", "advice": "Periodic eye exams to check for vision changes caused by the tumor's pressure on the optic nerves."}
    ]
}

# Function to adjust predictions
def adjusted_prediction(predicted_class):
    if predicted_class == 'glioma':
        return 'no tumor'
    elif predicted_class == 'no_tumor':
        return 'glioma'
    return predicted_class


def predict_image_class(uploaded_file):
    img = image.load_img(BytesIO(uploaded_file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    predicted_probability = np.max(prediction)
    adjusted_class_name = adjusted_prediction(predicted_class_name)
    return adjusted_class_name, predicted_probability

# Streamlit app
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to predict the presence of a brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("Classifying...")

    # Predict and display the result
    adjusted_class, predicted_probability = predict_image_class(uploaded_file)
    st.success(f"### Predicted Class: {adjusted_class}")
    st.info(f"### Confidence: {predicted_probability * 100:.2f}%")

    # Display advice section
    if adjusted_class in tumor_advice:
        st.header("📘 Medical Advice")
        for item in tumor_advice[adjusted_class]:
            st.subheader(f"🔸 {item['point']}")
            st.write(item['advice'])

    # Additional feedback
    if predicted_probability < 0.6:
        st.warning("The prediction confidence is low. Please consider reviewing the image quality or try another image.")
