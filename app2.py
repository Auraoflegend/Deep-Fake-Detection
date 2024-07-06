import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

# Load your model
model = tf.keras.models.load_model('models/deepfakemodel.h5')

# Function to extract frames from video
def extract_frames(video_path, num_frames=20):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    
    for i in range(0, total_frames, step):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len(frames) >= num_frames:
            break
    vidcap.release()
    return frames

# Function to process frames and make predictions
def process_frames(frames):
    predictions = []
    for frame in frames:
        resize = tf.image.resize(frame, (256, 256))
        yhat = model.predict(np.expand_dims(resize / 255.0, 0))
        predictions.append(yhat[0][0])
    return predictions

# Streamlit app
st.title('Deepfake Detection App')

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name
    
    # Extract frames from the video
    frames = extract_frames(temp_video_path)
    
    # Process the frames and make predictions
    predictions = process_frames(frames)
    
    # Calculate the average prediction
    avg_prediction = np.mean(predictions)
    
    if avg_prediction > 0.5:
        prediction_text = "<h2 style='text-align: center; color: green;'>Predicted video is original</h2>"
    else:
        prediction_text = "<h2 style='text-align: center; color: red;'>Predicted video is deepfake</h2>"
    
    
    st.markdown(prediction_text, unsafe_allow_html=True)
    
    
    # Display the original video
    st.video(uploaded_file)
    

