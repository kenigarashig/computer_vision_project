import streamlit as st
import numpy as np 
import tensorflow as tf
from PIL import Image
import io
from tensorflow.keras.models import load_model

# load the model
model=load_model("my_model_resnet50.keras")

def image_preprocess(image):
    img_size = (224, 224)
    image=image.resize(img_size)
    image =image.convert("RGB")
    image = np.array(image)  
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title("Parking Spot Detection")

    st.write("Upload an image of a parking spot, and the model will determine if it is empty or not.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = image_preprocess(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)

        # Interpret the result
        result = "Empty" if prediction[0][0] < 0.9999999 else "Not Empty"
        
        # Display the result
        st.write(f"Prediction: The parking spot is {result}.")
        st.write(prediction)

main()
