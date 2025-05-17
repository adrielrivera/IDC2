import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import serial

load_dotenv()
api_key = os.getenv("ROBOFLOW_API") 

# --- Serial Communication Setup ---
SERIAL_PORT = "/dev/ttyUSB0"  # <-- !!! REPLACE WITH YOUR ACTUAL PORT !!!
BAUD_RATE = 9600
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Successfully connected to Arduino on port {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
    print("Please check the port name and ensure Arduino is connected.")
    ser = None # Ensure ser is None if connection failed
# --------------------------------

# Explicitly start window thread
cv2.startWindowThread()

# Create window first before capturing
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Initialize Roboflow model
print("Initializing Roboflow model...")
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("idc2")
model = project.version("13").model
print("Model initialized!")

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)  # USB camera index 0

# Set resolution (lower for better performance)
resW, resH = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Set buffer size to minimum
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera ready!")

# Set bounding box colors
bbox_colors = [(0, 255, 0)]  # Green for bounding boxes

# Resize settings for detection (smaller is faster)
detection_width, detection_height = 320, 240

# Display instructions
print("Press 'd' to run detection on the current frame")
print("Press 'q' to quit")

# Global flag to store detection result
detection_result = None
detection_lock = threading.Lock()

def run_detection(frame):
    """Threaded detection function"""
    global detection_result
    
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (detection_width, detection_height))

    # Run prediction directly on the resized frame
    predictions = model.predict(resized_frame, confidence=40, overlap=30).json()

    # Acquire lock to safely update global detection result
    with detection_lock:
        detection_result = (predictions, resized_frame)
    
    # Send detected classes to Arduino if serial is available
    if ser and ser.is_open and predictions.get('predictions'):
        detected_classes = set() # Use a set to send each class name once per frame
        for p in predictions['predictions']:
            detected_classes.add(p['class'])
        for cls_name in detected_classes:
            command_to_send = f"DETECTED_{cls_name.upper()}\n"
            ser.write(command_to_send.encode('utf-8'))
            print(f"Sent to Arduino: {command_to_send.strip()}")
            time.sleep(0.05) # Small delay between commands if sending multiple

            # ---- ADDED: Read echo from Arduino ----
            # Wait a brief moment for Arduino to process and reply
            time.sleep(0.1) # Adjust if needed
            if ser.in_waiting > 0: # Check if there's data to read
                try:
                    echo_response = ser.readline().decode('utf-8').strip()
                    print(f"Received from Arduino: {echo_response}")
                except Exception as e:
                    print(f"Error reading from Arduino: {e}")
            # ---- END OF ADDED CODE ----

    print(f"Detection complete - Objects found: {len(predictions.get('predictions', []))}")

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print('Unable to read frames from the camera. Camera may be disconnected. Exiting program.')
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # If there is a detection result, draw it
        with detection_lock:
            if detection_result:
                predictions, detection_frame = detection_result
                for prediction in predictions['predictions']:
                    x1 = int(prediction['x'] - prediction['width'] / 2)
                    y1 = int(prediction['y'] - prediction['height'] / 2)
                    x2 = int(prediction['x'] + prediction['width'] / 2)
                    y2 = int(prediction['y'] + prediction['height'] / 2)

                    # Draw bounding box
                    color = bbox_colors[0]
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), color, 2)

                    # Add label
                    label = f"{prediction['class']}: {int(prediction['confidence']*100)}%"
                    cv2.putText(detection_frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Display detection results
                cv2.imshow('Detection Results', detection_frame)

                # Save the frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f'detected_frame_{timestamp}.png'
                cv2.imwrite(save_name, detection_frame)
                print(f"Frame saved as: {save_name}")

                # Clear the detection result after displaying
                detection_result = None

        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):  
            print("\n--- Running detection on current frame ---")
            threading.Thread(target=run_detection, args=(frame,)).start()

except KeyboardInterrupt:
    print("\nStopping detection...")
except Exception as e:
    print(f"Error in main loop: {str(e)}")

# Clean up
cap.release()
cv2.destroyAllWindows()
# --- Close Serial Port --- #
if ser and ser.is_open:
    ser.close()
    print("Serial port closed.")
# -------------------------
print("Done!")
