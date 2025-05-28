import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
# from datetime import datetime # Not used
import serial

# Initialize serial connection
def connect_to_serial():
    possible_ports = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1', '/dev/ttyUSB1']
    BAUD_RATE = 9600
    for port in possible_ports:
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1, write_timeout=1)
            print(f"Successfully connected to Arduino on port {port}")
            return ser
        except serial.SerialException as e:
            print(f"Info: Could not open serial port {port}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred trying port {port}: {e}")
    print("Error: Could not connect to Arduino on any specified ports.")
    return None

# Global variables
running = True
detection_active = False

def run_detection(frame, model_to_use, ser_port):
    global detection_active
    try:
        if model_to_use is None:
            print("CRITICAL THREAD ERROR: model_to_use is None. Cannot predict.")
            return

        resized_frame = cv2.resize(frame, (320, 240))
        predictions_data = model_to_use.predict(resized_frame, confidence=40, overlap=30).json()
        
        # Initialize counters for specific items
        syringe_count = 0
        bandage_count = 0
        gauze_count = 0

        # Process predictions to count specific items
        if predictions_data.get('predictions'):
            for p in predictions_data['predictions']:
                class_name = p['class'].upper() # Convert to uppercase for consistent matching
                if class_name == "SYRINGE":
                    syringe_count += 1
                elif class_name == "BANDAGE":
                    bandage_count += 1
                elif class_name == "GAUZE":
                    gauze_count += 1
                # Add other specific items here if needed in the future
            
            print(f"Counts: Syringes={syringe_count}, Bandages={bandage_count}, Gauzes={gauze_count}")

            # Send counts to Arduino if serial is available
            if ser_port and ser_port.is_open:
                counts_to_send = {
                    "SYRINGE": syringe_count,
                    "BANDAGE": bandage_count,
                    "GAUZE": gauze_count
                }
                
                total_items_of_interest_detected = syringe_count + bandage_count + gauze_count

                if total_items_of_interest_detected > 0:
                    for item_name, count in counts_to_send.items():
                        command_to_send = f"COUNT_{item_name}:{count}\n"
                        ser_port.write(command_to_send.encode('utf-8'))
                        print(f"Sent to Arduino: {command_to_send.strip()}")
                        time.sleep(0.05) # Small delay between messages
                        # Optional: Read echo if Arduino sends one per count message
                        # if ser_port.in_waiting > 0:
                        #     try:
                        #         echo_response = ser_port.readline().decode('utf-8').strip()
                        #         print(f"Received echo for count: {echo_response}")
                        #     except Exception as e:
                        #         print(f"Error reading echo for count: {e}")
                    # Send a general "detection processed" message
                    ser_port.write(b"DETECTION_PROCESSED\n")
                    print("Sent to Arduino: DETECTION_PROCESSED")
                else:
                    # If none of the target items were found, send a specific message
                    ser_port.write(b"NO_TARGET_ITEMS_FOUND\n")
                    print("Sent to Arduino: NO_TARGET_ITEMS_FOUND")
                    
        elif not predictions_data.get('predictions'):
            print("Detection run, no predictions returned by model for this frame.")
            if ser_port and ser_port.is_open:
                ser_port.write(b"NO_PREDICTIONS_FROM_MODEL\n") # New message for this case
                print("Sent to Arduino: NO_PREDICTIONS_FROM_MODEL")

        print(f"Detection complete - Total objects reported by model: {len(predictions_data.get('predictions', []))}")

    except AttributeError as ae:
        print(f"ATTRIBUTE ERROR in detection thread (model might be None or invalid): {ae}")
    except Exception as e:
        print(f"GENERAL ERROR in detection thread: {e}")
    finally:
        detection_active = False

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Exiting.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera initialized.")

# Initialize Roboflow for Medical Supplies
print("Initializing Roboflow model for Medical Supplies ('green-bean')...")
model_medical = None
ROBOFLOW_MODEL_VERSION_TO_LOAD = 2 # Assuming version 1 is the working one

try:
    rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")
    project_medical = rf.workspace().project("green-bean-5uqkj") # Corrected project name if needed
    print(f"Attempting to load model version: {ROBOFLOW_MODEL_VERSION_TO_LOAD}")
    model_medical = project_medical.version(ROBOFLOW_MODEL_VERSION_TO_LOAD).model
    
    if model_medical is not None:
        print(f"Medical Supplies Model ('green-bean', version {ROBOFLOW_MODEL_VERSION_TO_LOAD}) appears to be loaded.")
        if hasattr(model_medical, 'predict') and callable(getattr(model_medical, 'predict')):
            print("Model object has a 'predict' method.")
        else:
            print("WARNING: Model object loaded BUT does NOT have a callable 'predict' method! This will cause errors.")
            model_medical = None
    else:
        print(f"Error: Roboflow model (version {ROBOFLOW_MODEL_VERSION_TO_LOAD}) object is None after attempt to load.")
except Exception as e:
    print(f"CRITICAL ERROR initializing Roboflow model: {e}")
    model_medical = None

if model_medical is None:
    print("EXITING: Medical Supplies model could not be loaded. Cannot continue.")
    exit()

# Main loop
ser = connect_to_serial()
if ser is None:
    print("WARNING: Serial connection to Arduino failed. Detection will run but commands won't be sent/received.")

try:
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Camera issue?")
            time.sleep(0.5)
            continue

        if ser and ser.is_open and ser.in_waiting > 0:
            command = ""
            try:
                command = ser.readline().decode('utf-8').strip()
            except Exception as e:
                print(f"Error reading serial command: {e}")
            
            if command:
                print(f"Received command from Arduino: {command}")
                if command == "REQUEST_DETECTION" and not detection_active:
                    print("Medical supply detection requested by Arduino")
                    if frame is not None and model_medical is not None:
                        detection_active = True
                        detection_thread = threading.Thread(target=run_detection, args=(frame.copy(), model_medical, ser))
                        detection_thread.daemon = True
                        detection_thread.start()
                    else:
                        print("ERROR: No frame or model not loaded, cannot start detection.")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit command received via key press.")
            running = False
        elif key == ord('d') and not detection_active:
            print("\n--- Running manual medical supply detection (d key) ---")
            if frame is not None and model_medical is not None:
                detection_active = True
                detection_thread = threading.Thread(target=run_detection, args=(frame.copy(), model_medical, ser))
                detection_thread.daemon = True
                detection_thread.start()
            else:
                print("ERROR: No frame or model not loaded for manual detection.")
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping via KeyboardInterrupt (Ctrl+C)...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    print("Cleaning up...")
    running = False
    time.sleep(0.5) 
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if cap and cap.isOpened():
        cap.release()
        print("Camera released.")
    print("medical_supply_detector.py finished.")