import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import serial

# Load environment variables
load_dotenv()
api_key = os.getenv("ROBOFLOW_API")

# Try to connect to Arduino (will fail gracefully if not connected)
arduino = None
for port in ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1', '/dev/ttyUSB1']:
    try:
        arduino = serial.Serial(port, 9600, timeout=1)
        print(f"Connected to Arduino on {port}")
        time.sleep(2)  # Wait for Arduino to reset
        break
    except:
        pass

if not arduino:
    print("WARNING: Arduino not connected. You can use keyboard controls instead:")
    print(" - Press 'd' to run detection")
    print(" - Press 'q' to quit")

# Explicitly start window thread - THIS WAS MISSING
cv2.startWindowThread()

# Create window first before capturing - THIS WAS MISSING
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Initialize Roboflow model
print("Initializing Roboflow model...")
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("idc2")
model = project.version("15").model  # USING VERSION 15 LIKE OG.TXT
print("Model initialized!")

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)  # USB camera index 0

# If camera doesn't open, try other indices
if not cap.isOpened():
    for camera_index in [1, 2, -1]:
        print(f"Trying camera index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Successfully opened camera {camera_index}")
            break
    
    if not cap.isOpened():
        print("ERROR: Could not open any camera")
        exit(1)

# Set resolution (lower for better performance)
resW, resH = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Set buffer size to minimum - THIS WAS MISSING
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera ready!")

# Set bounding box colors
bbox_colors = [(0, 255, 0)]  # Green for bounding boxes

# Resize settings for detection (smaller is faster) - THIS WAS MISSING
detection_width, detection_height = 320, 240

# Global flag to store detection result - THIS WAS MISSING
detection_result = None
detection_lock = threading.Lock()

def run_detection(frame):
    """Threaded detection function"""
    global detection_result
    
    # Resize frame for faster processing - THIS WAS MISSING
    resized_frame = cv2.resize(frame, (detection_width, detection_height))

    # Run prediction directly on the resized frame - THIS WAS DIFFERENT
    predictions = model.predict(resized_frame, confidence=40, overlap=30).json()

    # Acquire lock to safely update global detection result
    with detection_lock:
        detection_result = (predictions, resized_frame)
    
    # Send results to Arduino if connected
    if arduino:
        send_results_to_arduino(predictions)
    
    print(f"Detection complete - Objects found: {len(predictions.get('predictions', []))}")

def send_results_to_arduino(predictions):
    """Send detection results to Arduino"""
    if not arduino:
        return
    
    try:
        # Start marker
        arduino.write(b"START\n")
        
        # Send count
        num_objects = len(predictions.get('predictions', []))
        arduino.write(f"COUNT:{num_objects}\n".encode())
        
        # Send each object
        if 'predictions' in predictions:
            for i, pred in enumerate(predictions['predictions']):
                x = pred['x'] / detection_width  # Normalize x to 0-1 range
                y = pred['y'] / detection_height  # Normalize y to 0-1 range
                class_name = pred['class']
                confidence = pred['confidence']
                
                object_info = f"ID:{i},X:{x:.2f},Y:{y:.2f},CLASS:{class_name},CONF:{confidence:.2f}\n"
                arduino.write(object_info.encode())
        
        # End marker
        arduino.write(b"END\n")
        print("Results sent to Arduino")
    
    except Exception as e:
        print(f"Error sending to Arduino: {str(e)}")

# Main loop
print("Detection server running.")
print("Press 'd' to run detection, 'q' to quit")

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print('Unable to read frames from the camera. Camera may be disconnected. Exiting program.')
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # If there is a detection result, draw it - THIS WAS DIFFERENT
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

        # Check for Arduino commands if connected
        if arduino and arduino.in_waiting > 0:
            command = arduino.readline().decode().strip()
            print(f"Received command: {command}")
            
            if command == "DETECT":
                # Capture frame
                ret, frame = cap.read()
                if ret:
                    # Run detection in a thread
                    print("\n--- Running detection from Arduino command ---")
                    threading.Thread(target=run_detection, args=(frame,)).start()
                else:
                    print("Error: Could not capture frame")
                    if arduino:
                        arduino.write(b"ERROR\n")
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):  # Manual detection trigger
            print("\n--- Running detection on current frame ---")
            threading.Thread(target=run_detection, args=(frame,)).start()

except KeyboardInterrupt:
    print("\nStopping detection...")
except Exception as e:
    print(f"Error in main loop: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")