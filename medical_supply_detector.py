import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
# from datetime import datetime # Not used in the provided test_server.py structure
import serial

# Initialize serial connection
def connect_to_serial():
    # Consider making this more robust by trying multiple ports
    # e.g., ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1', '/dev/ttyUSB1']
    SERIAL_PORT = "/dev/ttyACM0" # Or "/dev/ttyUSB0", etc. - check your Pi
    BAUD_RATE = 9600
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, write_timeout=1)
        print(f"Successfully connected to Arduino on port {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred trying port {SERIAL_PORT}: {e}")
        return None


# Global variables
running = True
detection_active = False
# detection_result = None # Not actively used in the provided test_server.py structure
# detection_lock = threading.Lock() # Not actively used for detection_result in test_server.py

def run_detection(frame, model_to_use, ser_port):
    """Threaded detection function"""
    global detection_active
    
    try:
        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (320, 240))

        # Run prediction
        # Confidence and overlap can be adjusted as needed
        predictions = model_to_use.predict(resized_frame, confidence=40, overlap=30).json()

        # Process predictions and send to Arduino
        if ser_port and ser_port.is_open and predictions.get('predictions'):
            detected_classes = set()
            for p in predictions['predictions']:
                detected_classes.add(p['class'])
            
            if not detected_classes:
                print("Detection run, but no objects met confidence/overlap.")
            
            for cls_name in detected_classes:
                command_to_send = f"DETECTED_{cls_name.upper().replace(' ', '_')}\n" # Replace spaces in class names if any
                ser_port.write(command_to_send.encode('utf-8'))
                print(f"Sent to Arduino: {command_to_send.strip()}")
                
                # Read echo from Arduino
                time.sleep(0.1) # Give Arduino a moment
                if ser_port.in_waiting > 0:
                    try:
                        echo_response = ser_port.readline().decode('utf-8').strip()
                        print(f"Received from Arduino: {echo_response}")
                    except Exception as e:
                        print(f"Error reading from Arduino: {e}")
        elif not predictions.get('predictions'):
            print("Detection run, no predictions returned by model.")


        print(f"Detection complete - Objects reported: {len(predictions.get('predictions', []))}")
        
    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        detection_active = False

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0) # Use 0 for default USB camera, or try other indices like 1, 2, -1
if not cap.isOpened():
    print("Error: Could not open camera. Exiting.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera initialized.")

# Initialize Roboflow for Medical Supplies
print("Initializing Roboflow model for Medical Supplies...")
try:
    rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU") # Assuming this is your general API key
    project_medical = rf.workspace().project("green-bean")
    model_medical = project_medical.version(1).model
    print("Medical Supplies Model ('green-bean', version 2) initialized!")
except Exception as e:
    print(f"Error initializing Roboflow model: {e}")
    print("Please check API key, project name, model version, and internet connection.")
    exit()


# Create windows (commented out for headless SSH operation)
# cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
# print("Window 'Camera Feed' prepared (but likely not shown in SSH).")

# Main loop
ser = connect_to_serial()

try:
    while running:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Camera issue?")
            time.sleep(0.5) # Wait a bit before retrying
            continue

        # Show frame (commented out for headless SSH operation)
        # cv2.imshow("Camera Feed", frame) 

        # Check for Arduino commands
        if ser and ser.is_open and ser.in_waiting > 0:
            command = ""
            try:
                command = ser.readline().decode('utf-8').strip()
            except Exception as e:
                print(f"Error reading serial command: {e}")
            
            if command: # If command was successfully read
                print(f"Received command from Arduino: {command}")
            
                if command == "REQUEST_DETECTION" and not detection_active:
                    print("Medical supply detection requested by Arduino")
                    detection_active = True
                    # Pass the correct model to the detection thread
                    detection_thread = threading.Thread(target=run_detection, args=(frame.copy(), model_medical, ser))
                    detection_thread.daemon = True
                    detection_thread.start()
                # Add other commands here if needed, e.g., "REQUEST_HOTDOG_DETECTION"
                # if you intend to run both models with one Python script later.
        
        # Handle key presses (less useful in headless mode without imshow, but q can still stop)
        key = cv2.waitKey(1) & 0xFF # cv2.waitKey is needed for some camera drivers to release frames
                                   # even if not showing images. It also provides a small delay.
        if key == ord('q'):
            print("Quit command received via key press.")
            running = False
        elif key == ord('d') and not detection_active:
            print("\n--- Running manual medical supply detection (d key) ---")
            if frame is not None:
                detection_active = True
                detection_thread = threading.Thread(target=run_detection, args=(frame.copy(), model_medical, ser))
                detection_thread.daemon = True
                detection_thread.start()
            else:
                print("No frame available for manual detection.")


        # Small delay to prevent CPU overuse if not much is happening
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping via KeyboardInterrupt (Ctrl+C)...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    print("Cleaning up...")
    running = False # Ensure all loops depending on 'running' will stop
    
    # Give threads a moment to finish if any were active
    # You might want more sophisticated thread joining if critical
    time.sleep(0.5) 

    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if cap and cap.isOpened():
        cap.release()
        print("Camera released.")
    
    # GUI cleanup (commented out for headless SSH)
    # cv2.destroyAllWindows()
    # for i in range(5):  # Ensure windows close properly (less relevant without imshow)
    #     cv2.waitKey(1)
    
    print("medical_supply_detector.py finished.")
