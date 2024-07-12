import streamlit as st
import onnxruntime as ort
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load the ONNX model
model_path = "handwriting_recognition_model.onnx"
session = ort.InferenceSession(model_path)

# Define the model input shape
input_shape = (128, 32, 3)  # Adjusted to 3 channels

# Function to preprocess the image
def preprocess_image(image):
    st.write("Original Image Shape:", image.shape)  # Debugging statement
    img = Image.fromarray(image.astype('uint8'), 'RGBA').convert('RGB')
    img = img.resize((input_shape[1], input_shape[0]))
    img = np.array(img)
    st.write("Resized Image Shape:", img.shape)  # Debugging statement
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    img = img.astype(np.float32)  # Ensure the data type is float32
    st.write("Preprocessed Image Shape:", img.shape)  # Debugging statement
    return img

# Function to predict the text from the image
def predict(image):
    preprocessed_img = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: preprocessed_img})
    prediction = result[0]
    st.write("Raw model output:", prediction)  # Debugging statement
    predicted_text, decoded_indices = decode_prediction(prediction)
    st.write("Decoded Indices:", decoded_indices)  # Debugging statement
    return predicted_text

# Improved function to decode the prediction
def decode_prediction(prediction, blank_index=0):
    """
    Decodes the prediction to a human-readable string using CTC decoding.

    Args:
        prediction (np.ndarray): The raw output from the model, a 2D array where each row is a softmax output for a character.
        blank_index (int): The index in the softmax output corresponding to the CTC blank token.

    Returns:
        str: The decoded string.
        list: Decoded indices.
    """
    decoded_text = []
    previous_char_index = -1
    decoded_indices = []

    for time_step in prediction[0]:
        char_index = np.argmax(time_step)
        if char_index != blank_index and char_index != previous_char_index:
            decoded_text.append(char_index)
            decoded_indices.append(char_index)
        previous_char_index = char_index

    # Adjust the character mapping to match your model's output
    char_mapping = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    decoded_string = ''.join(char_mapping[index - 1] if 0 < index <= len(char_mapping) else '' for index in decoded_text)
    return decoded_string, decoded_indices

# Streamlit interface
st.title("Handwriting Recognition")
st.write("Draw a word and click 'Predict' to see the prediction.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=128,
    width=32 * 16,  # Adjust width according to max text length
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    if st.button("Predict"):
        prediction = predict(img)
        st.write(f"Prediction: {prediction}")
