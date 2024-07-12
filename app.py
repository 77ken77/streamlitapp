# Adjust the character mapping to match your model's output
char_mapping = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def decode_prediction(prediction, blank_index=0):
    """
    Decodes the prediction to a human-readable string using CTC decoding.

    Args:
        prediction (np.ndarray): The raw output from the model, a 2D array where each row is a softmax output for a character.
        blank_index (int): The index in the softmax output corresponding to the CTC blank token.

    Returns:
        str: The decoded string.
    """
    decoded_text = []
    previous_char_index = -1

    for time_step in prediction[0]:
        char_index = np.argmax(time_step)
        if char_index != blank_index and char_index != previous_char_index:
            decoded_text.append(char_index)
        previous_char_index = char_index

    # Convert indices to characters using the char_mapping
    decoded_string = ''.join(char_mapping[index - 1] if 0 < index <= len(char_mapping) else '' for index in decoded_text)
    st.write("Decoded Indices:", decoded_text)  # Debugging statement
    return decoded_string
