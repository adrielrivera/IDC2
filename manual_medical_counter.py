import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np

# Global variables
running = True
detection_active = False # To prevent multiple detections at once

def count_medical_supplies(frame, model_to_use):
    """Threaded detection and counting function"""
    global detection_active
    try:
        if model_to_use is None:
            print("CRITICAL THREAD ERROR: model_to_use is None. Cannot predict.")
            return
        if frame is None:
            print("CRITICAL THREAD ERROR: frame is None. Cannot predict.")
            return

        print("Processing frame for medical supplies...")
        resized_frame = cv2.resize(frame, (320, 240))
        predictions_data = model_to_use.predict(resized_frame, confidence=40, overlap=30).json()
        
        syringe_count = 0
        bandage_count = 0
        gauze_count = 0

        if predictions_data.get('predictions'):
            for p in predictions_data['predictions']:
                class_name = p['class'].upper()
                if class_name == "SYRINGE":
                    syringe_count += 1
                elif class_name == "BANDAGE":
                    bandage_count += 1
                elif class_name == "GAUZE":
                    gauze_count += 1
            
            print("--- Detection Counts ---")
            print(f"  Syringes: {syringe_count}")
            print(f"  Bandages: {bandage_count}")
            print(f"  Gauzes  : {gauze_count}")
            print(f"  (Total objects reported by model: {len(predictions_data.get('predictions', []))})")
            print("------------------------")

        else:
            print("--- Detection Counts ---")
            print("  No predictions returned by model for this frame.")
            print("------------------------")

    except AttributeError as ae:
        print(f"ATTRIBUTE ERROR in detection thread: {ae}")
    except Exception as e:
        print(f"GENERAL ERROR in detection thread: {e}")
    finally:
        detection_active = False
        print("Detection thread finished.")

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
ROBOFLOW_MODEL_VERSION_TO_LOAD = 1 # Make sure this is the correct, working version

try:
    # IMPORTANT: Replace with your actual API key if different
    rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU") 
    # IMPORTANT: Replace with your exact Roboflow project ID for "green-bean"
    # e.g., "green-bean-XXXXX" if it has a unique ID suffix
    project_medical = rf.workspace().project("green-bean") 
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

print("\nPython Medical Supply Counter Ready.")
print("Press 'c' to count items in the current camera view.")
print("Press 'q' to quit.")
print("Ensure this SSH window is active to capture key presses.")

# --- Main Loop ---
# You can uncomment cv2.imshow lines if you are using X11 forwarding to see the feed on your Mac
# cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL) # Uncomment for X11 forwarding

try:
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Camera issue?")
            time.sleep(0.5)
            continue

        # cv2.imshow("Camera Feed", frame) # Uncomment for X11 forwarding

        key = cv2.waitKey(1) & 0xFF # Essential for imshow to work and to capture keys
        
        if key == ord('q'):
            print("Quit command received.")
            running = False
        elif key == ord('c') and not detection_active: # 'c' for Count
            print("\n'c' pressed - Starting medical supply count...")
            if frame is not None and model_medical is not None:
                detection_active = True
                # Make sure to pass a copy of the frame if the original is still being used/displayed
                detection_thread = threading.Thread(target=count_medical_supplies, args=(frame.copy(), model_medical))
                detection_thread.daemon = True
                detection_thread.start()
            else:
                print("ERROR: No frame available or model not loaded for counting.")
        
        # If not using imshow, a small sleep can prevent this loop from consuming 100% CPU
        # if cv2.waitKey(1) is not sufficient or if imshow is commented out.
        # However, cv2.waitKey(1) already provides a small delay.
        # time.sleep(0.01) # Potentially add if CPU usage is too high without imshow

except KeyboardInterrupt:
    print("\nStopping via KeyboardInterrupt (Ctrl+C)...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    print("Cleaning up...")
    running = False
    time.sleep(0.5) # Give threads a chance to finish
    if cap and cap.isOpened():
        cap.release()
        print("Camera released.")
    # cv2.destroyAllWindows() # Uncomment if you were using cv2.imshow
    print("Medical supply counter script finished.")