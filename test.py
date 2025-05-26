import cv2
import time
from roboflow import Roboflow
import os
import serial

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

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)

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

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera ready!")

# Initialize Roboflow model - EXACT OG.TXT APPROACH
print("Initializing Roboflow model...")
rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")  # Use the hardcoded key from og.txt
project = rf.workspace().project("idc2")
model = project.version("14").model  # Make sure version matches og.txt
print("Model initialized!")

# Temp file for saving frames
temp_file = "temp_frame.jpg"

def run_detection(frame):
    """Run object detection on the provided frame"""
    print("Running detection...")
    
    try:
        # Save frame temporarily for Roboflow - EXACT APPROACH FROM OG.TXT
        cv2.imwrite(temp_file, frame)
        
        # Run prediction exactly as in og.txt
        predictions = model.predict(temp_file, confidence=40, overlap=30).json()
        print(f"Received {len(predictions.get('predictions', []))} detections")
        
        # Display the detections on the frame
        result_frame = frame.copy()
        if 'predictions' in predictions:
            for pred in predictions['predictions']:
                x1 = int(pred['x'] - pred['width'] / 2)
                y1 = int(pred['y'] - pred['height'] / 2)
                x2 = int(pred['x'] + pred['width'] / 2)
                y2 = int(pred['y'] + pred['height'] / 2)
                
                # Draw bounding box and label
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{pred['class']}: {int(pred['confidence']*100)}%"
                cv2.putText(result_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Print detection details
                print(f"Detected {pred['class']} with {pred['confidence']:.2f} confidence")
        
        # Display results
        cv2.imshow("Detection Result", result_frame)
        cv2.waitKey(1)
        
        return predictions
    
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"predictions": []}

def send_results_to_arduino(predictions):
    """Send detection results to Arduino"""
    if not arduino:
        print("Arduino not connected, cannot send results")
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
                x = pred['x'] / 640  # Normalize x to 0-1 range
                y = pred['y'] / 480  # Normalize y to 0-1 range
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
if not arduino:
    print("Use 'd' key to trigger detection manually")

try:
    while True:
        # Check for Arduino commands if connected
        if arduino and arduino.in_waiting > 0:
            command = arduino.readline().decode().strip()
            print(f"Received command: {command}")
            
            if command == "DETECT":
                # Capture frame
                ret, frame = cap.read()
                if ret:
                    # Run detection
                    start_time = time.time()
                    predictions = run_detection(frame)
                    detection_time = time.time() - start_time
                    print(f"Detection completed in {detection_time:.2f} seconds")
                    
                    # Send results back to Arduino
                    send_results_to_arduino(predictions)
                else:
                    print("Error: Could not capture frame")
                    if arduino:
                        arduino.write(b"ERROR\n")
        
        # Display camera feed when not detecting
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Feed", frame)
        
        # Check for keyboard input
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):  # Manual detection trigger
            if ret:
                start_time = time.time()
                predictions = run_detection(frame)
                detection_time = time.time() - start_time
                print(f"Detection completed in {detection_time:.2f} seconds")
                
                # Print detailed detection results
                if 'predictions' in predictions and len(predictions['predictions']) > 0:
                    print("\nDetected objects:")
                    for i, pred in enumerate(predictions['predictions']):
                        print(f"  {i+1}. {pred['class']} ({pred['confidence']*100:.1f}%)")
                else:
                    print("No objects detected")
            else:
                print("Error: Could not capture frame")
        
        # Small delay to prevent high CPU usage
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Shutting down...")
except Exception as e:
    print(f"Error in main loop: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")