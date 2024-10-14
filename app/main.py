import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Corrected model path
# model_path = "\app\trained_model\Fruit_and_Vegetable_Diseases.h5"
model_path = "C:\\Users\\Shanu\\Desktop\\shampro\\app\\trained_model\\Fruit_and_Vegetable_Diseases.h5"

# Print the model path for debugging
print(f"Loading model from: {model_path}")

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except FileNotFoundError as e:
    print(e)
    raise

# Loading the class names
class_indices_path = os.path.join(working_dir, "class_indices.json")
class_indices = json.load(open(class_indices_path))

# ... rest of your code remains the same ...


# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    img = image.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
