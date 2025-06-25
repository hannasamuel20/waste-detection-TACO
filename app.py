import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.title("Waste Sorting Detection")
uploaded_file = st.file_uploader("Upload conveyor image", type=["jpg", "png"])
if uploaded_file:
    model = YOLO('runs/detect/taco_yolo8m/weights/best.pt')
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(image)
    st.image(results[0].plot(), caption="Detected Waste", use_column_width=True)