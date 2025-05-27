import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import serial

# Initialize serial connection
def connect_to_serial():
    SERIAL_PORT = "/dev/ttyUSB0"  # Change if needed
    BAUD_RATE = 9600
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Successfully connected to Arduino on port {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
        return None

# Explicitly start window thread
cv2.startWindowThread()

# Create window first before capturing
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Initialize Roboflow model
print("Initializing Roboflow model...")
rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")
project = rf.workspace().project("idc2")
model = project.version("15").model
print("Model initialized!")

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)

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

# Global flag to store detection result
detection_result = None
detection_lock = threading.Lock()

def run_detection(frame, ser):
    """Threaded detection function"""
    global detection_result
    
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (detection_width, detection_height))

    # Run prediction directly on the resized frame
    predictions = model.predict(resized_frame, confidence=40, overlap=30).json()

    # Acquire lock to safely update global detection result
    with detection_lock:
        detection_result = (predictions, resized_frame)
    
    # Draw predictions on the frame
    for prediction in predictions.get('predictions', []):
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)

        # Draw bounding box
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"{prediction['class']}: {int(prediction['confidence']*100)}%"
        cv2.putText(resized_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show detection frame
    cv2.imshow('Detection Results', resized_frame)
    
    # Send detected classes to Arduino if serial is available
    if ser and ser.is_open and predictions.get('predictions'):
        detected_classes = set() # Use a set to send each class name once per frame
        for p in predictions['predictions']:
            detected_classes.add(p['class'])
        for cls_name in detected_classes:
            command_to_send = f"DETECTED_{cls_name.upper()}\n"
            ser.write(command_to_send.encode('utf-8'))
            print(f"Sent to Arduino: {command_to_send.strip()}")
            time.sleep(0.05) # Small delay between commands
            
            # Read echo from Arduino
            time.sleep(0.1)
            if ser.in_waiting > 0:
                try:
                    echo_response = ser.readline().decode('utf-8').strip()
                    print(f"Received from Arduino: {echo_response}")
                except Exception as e:
                    print(f"Error reading from Arduino: {e}")
    
    print(f"Detection complete - Objects found: {len(predictions.get('predictions', []))}")

# Main loop
ser = connect_to_serial()

try:
    while True:
        # Always capture and show frame
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Feed', frame)
        else:
            print("Failed to capture frame")
            continue

        # Check for Arduino commands
        if ser and ser.in_waiting > 0:
            command = ser.readline().decode('utf-8').strip()
            print(f"Received command: {command}")
            
            if command == "REQUEST_DETECTION":
                print("Detection requested by Arduino")
                if ret:
                    threading.Thread(target=run_detection, args=(frame.copy(), ser)).start()
                else:
                    print("Failed to capture frame for detection")
                    ser.write("ERROR\n".encode('utf-8'))

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            print("\n--- Running manual detection on current frame ---")
            if ret:
                threading.Thread(target=run_detection, args=(frame.copy(), ser)).start()

        time.sleep(0.01)  # Small delay to prevent CPU overuse

except KeyboardInterrupt:
    print("\nStopping detection...")
except Exception as e:
    print(f"Error in main loop: {str(e)}")
finally:
    # Clean up
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Additional wait to ensure windows close
    print("Done!")