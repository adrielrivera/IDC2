import cv2
import os
import time
from roboflow import Roboflow
import threading
import numpy as np
import sys # For sys.stdin
import select # For non-blocking input
import tty # For raw terminal mode
import termios # For terminal attributes

# Global variables
running = True
detection_active = False # To prevent multiple detections at once

# Store original terminal settings
original_termios_settings = None

def set_tty_cbreak(fd):
    """Set terminal to cbreak mode (non-canonical, no echo)"""
    global original_termios_settings
    original_termios_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def restore_tty_settings(fd):
    """Restore original terminal settings"""
    if original_termios_settings:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_termios_settings)

def is_data_available():
    """Check if there's data available to read on stdin"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

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

        print("\nProcessing frame for medical supplies...") # Newline for better formatting
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
        print("Press 'c' to count again, or 'q' to quit.") # Reminder

    except AttributeError as ae:
        print(f"ATTRIBUTE ERROR in detection thread: {ae}")
    except Exception as e:
        print(f"GENERAL ERROR in detection thread: {e}")
    finally:
        detection_active = False
        # print("Detection thread finished.") # Can be a bit noisy

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
ROBOFLOW_MODEL_VERSION_TO_LOAD = 2

try:
    rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU") 
    project_medical = rf.workspace().project("green-bean-5uqkj") 
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
print("Press 'c' (and Enter if needed by your terminal) to count items.")
print("Press 'q' (and Enter if needed by your terminal) to quit.")
print("Ensure this SSH window is active.")

# --- Main Loop ---
# No cv2.imshow() or cv2.namedWindow() for headless operation

# Set terminal to cbreak mode to read single characters if possible
# This is more complex and can leave terminal in a weird state if script crashes
# For simplicity with SSH, we might rely on Enter key after 'c' or 'q'
# If you want true single-key press without Enter, tty/termios is needed:
# fd = sys.stdin.fileno()
# set_tty_cbreak(fd)

try:
    while running:
        ret, frame = cap.read()
        if not ret:
            # print("Failed to capture frame. Camera issue?") # Can be noisy
            time.sleep(0.1) # Give camera a moment if it failed
            continue

        # Check for keyboard input without blocking
        if is_data_available():
            key = sys.stdin.read(1) # Read a single character
            
            if key == 'q':
                print("Quit command 'q' received.")
                running = False
            elif key == 'c' and not detection_active:
                print("\n'c' received - Starting medical supply count...")
                if frame is not None and model_medical is not None:
                    detection_active = True
                    detection_thread = threading.Thread(target=count_medical_supplies, args=(frame.copy(), model_medical))
                    detection_thread.daemon = True
                    detection_thread.start()
                else:
                    print("ERROR: No frame available or model not loaded for counting.")
        
        time.sleep(0.05) # Main loop delay, adjust as needed

except KeyboardInterrupt:
    print("\nStopping via KeyboardInterrupt (Ctrl+C)...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    print("Cleaning up...")
    running = False
    # restore_tty_settings(fd) # Restore terminal if set_tty_cbreak was used
    time.sleep(0.5) 
    if cap and cap.isOpened():
        cap.release()
        print("Camera released.")
    print("Medical supply counter script finished.")