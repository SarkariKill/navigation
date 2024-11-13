import cv2
import torch
import streamlit as st
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Create Streamlit UI elements
st.title("Object Detection with Directional Feedback")
frame_placeholder = st.empty()

# Initialize camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define custom box dimensions (bottom-center of the frame)
rect_width_percent = 80  # Width of rectangle as % of frame width
rect_height_percent = 30  # Height of rectangle as % of frame height
rect_width = int((rect_width_percent / 100) * frame_width)
rect_height = int((rect_height_percent / 100) * frame_height)
x_start = (frame_width - rect_width) // 2  # Center horizontally
x_end = x_start + rect_width
y_end = frame_height  # Bottom edge
y_start = y_end - rect_height  # Start of rectangle height

# Define boundaries for sections within the custom box
left_boundary = x_start + rect_width // 3
right_boundary = x_start + 2 * (rect_width // 3)

# Streamlit button to stop the video feed
stop_button = st.button("Stop Camera")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the custom box on the frame
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.line(frame, (left_boundary, y_start), (left_boundary, y_end), (0, 255, 0), 2)
    cv2.line(frame, (right_boundary, y_start), (right_boundary, y_end), (0, 255, 0), 2)
    cv2.putText(frame, "LEFT", (x_start + 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "MIDDLE", (left_boundary + 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "RIGHT", (right_boundary + 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Perform object detection
    results = model(frame)

    # Loop through detected objects and check if they are in the custom box area
    for result in results.xyxy[0]:  # `results.xyxy[0]` is a tensor of detections
        x_min, y_min, x_max, y_max, confidence, cls = map(int, result.tolist()[:6])
        label = model.names[cls]
        center_x = (x_min + x_max) // 2  # Center x-coordinate of the object
        center_y = (y_min + y_max) // 2  # Center y-coordinate of the object

        # Check if the detected object's center is within the custom box
        if y_start <= center_y <= y_end and x_start <= center_x <= x_end:
            # Determine section within the custom box and display feedback
            if center_x < left_boundary:
                feedback_text = f"{label} detected on your left."
                cv2.putText(frame, feedback_text, (x_start + 10, y_start - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif center_x < right_boundary:
                feedback_text = f"{label} detected in the middle."
                cv2.putText(frame, feedback_text, (left_boundary + 10, y_start - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                feedback_text = f"{label} detected on your right."
                cv2.putText(frame, feedback_text, (right_boundary + 10, y_start - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Convert the frame to RGB for Streamlit display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb_frame, channels="RGB")

cap.release()
