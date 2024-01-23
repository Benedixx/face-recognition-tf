# Import packages
import os
import cv2
import numpy as np
import sys
import random
from tensorflow.lite.python.interpreter import Interpreter

# Load the label map into memory
def load_labels(lblpath):
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Load the Tensorflow Lite model into memory
def load_model(modelpath):
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    return interpreter, height, width, input_details, output_details

# Define function for webcam inferencing
def tflite_detect_webcam(modelpath, lblpath, min_conf=0.5, txt_only=False):
    # Load label map and TFLite model
    labels = load_labels(lblpath)
    interpreter, height, width, input_details, output_details = load_model(modelpath)

    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break

        # Resize frame to match model's expected sizing
        input_data = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(input_data, axis=0)

        # Normalize pixel values if using a floating model
        if input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if scores[i] > min_conf:
                ymin, xmin, ymax, xmax = boxes[i]  # Bounding box coordinates
                ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                label_ymin = max(ymin, 10)  # Make sure not to draw label too close to top of window
                cv2.putText(frame, label, (xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Webcam Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Set up variables for running user's model
PATH_TO_MODEL = 'custom_model_lite/detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS = 'custom_model_lite/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold = 0.5   # Confidence threshold

# Run webcam inferencing function
tflite_detect_webcam(PATH_TO_MODEL, PATH_TO_LABELS, min_conf_threshold)
