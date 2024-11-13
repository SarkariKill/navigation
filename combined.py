import streamlit as st
import cv2
from ultralytics import YOLO
import threading

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Function to calculate distance in cm using the pinhole camera model
def calculate_distance_cm(focal_length, real_object_width, object_width_in_image):
    return (real_object_width * focal_length) / object_width_in_image

# Real object widths in centimeters (for common objects)
object_widths_cm = {
    'person': 45,          # Approximate width of a person in cm
    'bicycle': 60,         # Standard bicycle width
    'car': 180,            # Width of a car
    'motorbike': 80,       # Width of a motorbike
    'dog': 30,             # Width of a dog
    'chair': 50,           # Average chair width
    'table': 80,           # Dining table width
    'bottle': 7,           # Width of a bottle
    'backpack': 30,        # Width of a backpack
    'sofa': 200,           # Width of a large sofa
    'laptop': 35,          # Width of a standard laptop
    'cup': 8,              # Width of a cup
    'tvmonitor': 90,       # Width of a typical TV monitor
    'book': 15,            # Width of a closed book
    'remote': 5,           # Width of a TV remote
    'apple': 8,            # Average width of an apple
    'umbrella': 100,       # Width of an open umbrella
    'pen': 1,              # Width of a pen
    'mouse': 6,            # Width of a computer mouse
    'keyboard': 45,        # Width of a keyboard
    'phone': 7,            # Width of a mobile phone
    'shoe': 10,            # Width of an average shoe
    'bowl': 15,            # Width of a bowl
    'microwave': 50,       # Width of a microwave oven
    'toaster': 30,         # Width of a toaster
    'refrigerator': 90,    # Width of a refrigerator
    'sink': 50,            # Width of a sink
    'fork': 2.5,           # Width of a fork
}

# Focal length in pixels (camera calibration)
focal_length = 700  # Focal length from your provided code

# Streamlit layout
st.title("Live YOLOv8 Object Detection with Central Region Focus")

# Checkbox to start/stop the camera
run = st.checkbox('Run Camera', value=False)

# Slider to control the size of the detection rectangle
rect_width_percent = st.slider("Central Rectangle Width (% of frame width)", 10, 100, 30)
rect_height_percent = st.slider("Central Rectangle Height (% of frame height)", 10, 100, 30)

# Placeholder for displaying video
FRAME_WINDOW = st.empty()

# Set up a threading lock to safely manage camera access
lock = threading.Lock()

if run:
    st.write("Camera is running...")
    cap = cv2.VideoCapture(0)  # Open the camera feed

    # Set camera resolution (optional: lower resolution for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera calibration constants for distance calculation
    frame_skip = 3  # Process every 3rd frame to improve performance
    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Skip frames to reduce computation load
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Calculate the dimensions of the central rectangle
        rect_width = int((rect_width_percent / 100) * frame_width)
        rect_height = int((rect_height_percent / 100) * frame_height)
        x_start = (frame_width - rect_width) // 2
        y_start = (frame_height - rect_height) // 2
        x_end = x_start + rect_width
        y_end = y_start + rect_height

        # Draw the central detection rectangle (for visualization)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)  # Blue rectangle

        # Perform object detection only within the central region
        central_region = frame[y_start:y_end, x_start:x_end]
        results = model(central_region)

        # Extract bounding boxes and labels for objects within the central region
        for result in results:
            for box in result.boxes:  # Iterate through detected objects
                label = model.names[int(box.cls)]  # Convert class index to label name
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

                # Calculate the bounding box position relative to the entire frame
                x_min += x_start
                y_min += y_start
                x_max += x_start
                y_max += y_start

                # Calculate width of the detected object in the image
                object_width_in_image = x_max - x_min

                # Check if the object label has a known real-world width
                if label in object_widths_cm:
                    real_object_width = object_widths_cm[label]
                    # Calculate distance to the object using the pinhole camera model (in cm)
                    distance_cm = calculate_distance_cm(focal_length, real_object_width, object_width_in_image)

                    # Print the distance to the object in the terminal
                    print(f'Distance to {label}: {distance_cm:.2f} cm')

                    # Display the bounding box and distance
                    # Set bounding box color to black (0, 0, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                    text = f'{label}: {distance_cm:.2f} cm'
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

    cap.release()  # Release the camera when done
else:
    st.write("Camera is stopped.")
