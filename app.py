import os
import io
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import gdown

# Set page configuration (MUST be at the top)
st.set_page_config(page_title="Waste Classification", page_icon="♻️")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define Google Drive File ID and Model Path
DRIVE_FILE_ID = "1Gyt0S0y3r8h6hE0-2CZL2iJb-DKGDQDR"  # Updated File ID for the model in Google Drive
MODEL_PATH = "model.tflite"  # Path to save/download the model locally

# Download model from Google Drive if it doesn't exist locally
def download_model():
    """Downloads the model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("Downloading model from Google Drive..."):
                url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            st.error("Failed to download the model. Please check the Drive link or file permissions.")
            st.stop()
    else:
        logging.info("Model already exists locally.")

# Load TFLite model
def load_model():
    """Loads the TensorFlow Lite model."""
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.info("TFLite model loaded successfully.")
        return interpreter, input_details, output_details
    except Exception as e:
        logging.error(f"Error loading TFLite model: {e}")
        st.error("Failed to load the model. Please check the model file or try again.")
        return None, None, None

# Preprocess image function
def preprocess_image(file, input_details):
    """
    Preprocesses the uploaded image for model inference.
    Converts the image to RGB, resizes it to the model's input shape, and normalizes pixel values.
    """
    try:
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB')
        input_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
        img = img.resize(input_shape)
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        return np.expand_dims(img_array, axis=0).astype(np.float32)
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        st.error("Error processing image. Please try again with a valid image file.")
        return None

# Model and file checks
download_model()
interpreter, input_details, output_details = load_model()

# Streamlit UI
st.title("♻️ Waste Classification using CNN")
st.write("Upload an image to classify it as either **'Organic'** or **'Recyclable'**.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

CLASS_NAMES = ['Organic', 'Recyclable']

if uploaded_file is not None:
    if interpreter is not None:
        img_array = preprocess_image(uploaded_file, input_details)

        if img_array is not None:
            with st.spinner("Classifying..."):  # Add a spinner while processing
                try:
                    interpreter.set_tensor(input_details[0]['index'], img_array)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])

                    predicted_class_index = np.argmax(prediction)
                    predicted_class = CLASS_NAMES[predicted_class_index]
                    predicted_score = prediction[0][predicted_class_index] * 100

                    col1, col2 = st.columns([1, 1])  # Equal columns for image and results
                    with col1:
                        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    with col2:
                        st.markdown(f"### **Prediction:** {predicted_class}")
                        st.markdown(f"### **Confidence:** {predicted_score:.2f}%")
                except Exception as e:
                    logging.error(f"Error during classification: {e}")
                    st.error("An error occurred while processing the image. Please try again.")
    else:
        st.error("Model is not loaded. Please restart the app or check the model file.")

st.markdown("---")  # Separator line
st.markdown("### 📚 About")
st.markdown(
    """
    This project uses a **Convolutional Neural Network (CNN)** model to classify waste into two categories:
    - **Organic**: Biodegradable materials like food waste, garden waste, etc.
    - **Recyclable**: Non-biodegradable materials like plastic, metal, paper, etc.
    
    Built using TensorFlow and Streamlit.
    """
)
