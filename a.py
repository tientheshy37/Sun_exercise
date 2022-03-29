import streamlit as st 
import cv2 
import keras
import tensorflow as tf 
import numpy as np 
from tensorflow.keras.utils import CustomObjectScope

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

st.header("Input the image")
uploaded_file = st.file_uploader("Choose a image file")

if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # print(type(file_bytes))
    opencv_image = cv2.imdecode(file_bytes, 1)
    # print(type(opencv_image))
    opencv_image = cv2.resize(opencv_image, (256, 256))
    opencv_image = opencv_image / 255.0
    st.image(opencv_image, channels="BGR")
    with CustomObjectScope({'iou': iou}):
        model = keras.models.load_model("model.h5")
    y_pred = model.predict(np.expand_dims(opencv_image, axis=0))[0] > 0.5
    # print(y_pred)
    # print(opencv_image)

    h, w, _ = opencv_image.shape
    white_line = np.ones((h, 10, 3)) * 255.0

    all_images = [
        opencv_image,
        white_line,
        mask_parse(y_pred)
    ]

    st.subheader("Result")
    image = np.concatenate(all_images, axis=1)
    st.image(image, clamp=True, channels="BGR")