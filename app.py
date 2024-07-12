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
    img = Image.fromarray(image.astype('uint8'), 'RGBA').convert('RGB')
    img = img.resize((input_shape[1], input_shape[0]))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    img = img.astype(np.float32)  # Ensure the data type is float32
    return img

# Function to predict the text from the image
def predict(image):
    preprocessed_img = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: preprocessed_img})
    prediction = result[0]
    st.write("Raw model output:", prediction)  # Debugging statement
    predicted_text = decode_prediction(prediction)
    return predicted_text

# Mock function to decode the prediction (Replace this with actual decoding logic)
def decode_prediction(prediction, blank_index=0):
    """
    Decodes the prediction to a human-readable string using CTC decoding.

    Args:
        prediction (np.ndarray): The raw output from the model, a 2D array where each row is a softmax output for a character.
        blank_index (int): The index in the softmax output corresponding to the CTC blank token.

    Returns:
        str: The decoded string.
    """
    predicted_text = []
    previous_char_index = -1  # To track and collapse repeated characters
    for char_probs in prediction[0]:
        char_index = np.argmax(char_probs)
        if char_index != blank_index and char_index != previous_char_index:
            predicted_text.append(char_index)
        previous_char_index = char_index

    # Convert indices to characters assuming 'A' starts at index 1
    decoded_text = ''.join(chr(index + ord('A') - 1) for index in predicted_text)
    return decoded_text



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
