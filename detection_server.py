import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
from datetime import datetime
import serial

# Initialize serial connection
def connect_to_serial():
    SERIAL_PORT = "/dev/ttyACM0"
    BAUD_RATE = 9600
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Successfully connected to Arduino on port {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
        return None

# Global variables
running = True
detection_active = False
detection_result = None
detection_lock = threading.Lock()

def run_detection(frame, ser):
    """Threaded detection function"""
    global detection_active, detection_result
    
    try:
        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (320, 240))

        # Run prediction
        predictions = model.predict(resized_frame, confidence=40, overlap=30).json()

        # Process predictions and send to Arduino
        if ser and ser.is_open and predictions.get('predictions'):
            detected_classes = set()
            for p in predictions['predictions']:
                detected_classes.add(p['class'])
            for cls_name in detected_classes:
                command_to_send = f"DETECTED_{cls_name.upper()}\n"
                ser.write(command_to_send.encode('utf-8'))
                print(f"Sent to Arduino: {command_to_send.strip()}")
                
                # Read echo from Arduino
                time.sleep(0.1)
                if ser.in_waiting > 0:
                    try:
                        echo_response = ser.readline().decode('utf-8').strip()
                        print(f"Received from Arduino: {echo_response}")
                    except Exception as e:
                        print(f"Error reading from Arduino: {e}")

        print(f"Detection complete - Objects found: {len(predictions.get('predictions', []))}")
        
    except Exception as e:
        print(f"Error in detection: {e}")
    finally:
        detection_active = False

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Initialize Roboflow
print("Initializing Roboflow model...")
rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")
project = rf.workspace().project("idc2")
model = project.version("15").model
print("Model initialized!")

# Create windows
# cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
# commented out because im running code via ssh

# Main loop
ser = connect_to_serial()

try:
    while running:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        # Show frame
        # commented out because im running code via ssh
        # cv2.imshow("Camera Feed", frame) 

        # Check for Arduino commands
        if ser and ser.in_waiting > 0:
            command = ser.readline().decode('utf-8').strip()
            print(f"Received command: {command}")
            
            if command == "REQUEST_DETECTION" and not detection_active:
                print("Detection requested by Arduino")
                detection_active = True
                detection_thread = threading.Thread(target=run_detection, args=(frame.copy(), ser))
                detection_thread.daemon = True
                detection_thread.start()

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('d') and not detection_active:
            print("\n--- Running manual detection ---")
            detection_active = True
            detection_thread = threading.Thread(target=run_detection, args=(frame.copy(), ser))
            detection_thread.daemon = True
            detection_thread.start()

        # Small delay to prevent CPU overuse
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    running = False
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed")
    cap.release()
    # cv2.destroyAllWindows()
    # for i in range(5):  # Ensure windows close properly
    #     cv2.waitKey(1)
    print("Done!")
    