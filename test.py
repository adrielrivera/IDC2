import cv2
import time
from roboflow import Roboflow
import os
import threading
import queue
import numpy as np
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ROBOFLOW_API')

# Queue for communication between threads
prediction_queue = queue.Queue()
latest_predictions = []

# Function to run predictions in a separate thread
def prediction_worker():
    while not stop_threads:
        try:
            if not frame_queue.empty():
                print("\n--- Starting new prediction ---")
                start_time = time.time()
                
                frame = frame_queue.get()
                print(f"Frame retrieved from queue: {time.time() - start_time:.2f} seconds")
                
                # Save frame temporarily for Roboflow
                cv2.imwrite(temp_file, frame)
                print(f"Frame saved to disk: {time.time() - start_time:.2f} seconds")
                
                # Run prediction on saved frame
                print("Starting Roboflow prediction...")
                predictions = model.predict(temp_file, confidence=40, overlap=30).json()
                print(f"Roboflow prediction completed: {time.time() - start_time:.2f} seconds")
                
                # Put predictions in the queue
                prediction_queue.put(predictions)
                print(f"Prediction added to queue: {time.time() - start_time:.2f} seconds")
                
                # Don't block if queue is full
                frame_queue.task_done()
                print(f"Total processing time: {time.time() - start_time:.2f} seconds")
                print("--- Prediction cycle complete ---\n")
                
        except Exception as e:
            print(f"Error in prediction thread: {str(e)}")
        
        # Sleep a bit to prevent CPU overload
        time.sleep(0.01)

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
resW, resH = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Set buffer size to minimum
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera ready!")

# Temp file for saving frames
temp_file = "temp_frame.jpg"

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

# Set bounding box colors (using the Tableau 10 color scheme)
bbox_colors = [(0, 255, 0)]  # Using green for now, can expand with more colors

# Thread control
stop_threads = False
frame_queue = queue.Queue(maxsize=1)  # Only store the latest frame

# Start prediction thread
prediction_thread = threading.Thread(target=prediction_worker)
prediction_thread.daemon = True
prediction_thread.start()

print("Starting detection loop...")
print("Press 'q' to quit, 's' to pause, 'p' to save current frame")

try:
    while True:
        t_start = time.perf_counter()

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print('Unable to read frames from the camera. Camera may be disconnected. Exiting program.')
            break

        # Run prediction on frame
        try:
            predictions = model.predict(frame, confidence=40, overlap=30).json()
            
            # Initialize variable for basic object counting
            object_count = 0

            # Draw predictions
            if 'predictions' in predictions:
                for prediction in predictions['predictions']:
                    # Get coordinates
                    x1 = prediction['x'] - prediction['width'] / 2
                    y1 = prediction['y'] - prediction['height'] / 2
                    x2 = prediction['x'] + prediction['width'] / 2
                    y2 = prediction['y'] + prediction['height'] / 2
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box
                    color = bbox_colors[0]  # Using first color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{prediction['class']}: {int(prediction['confidence']*100)}%"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(y1, labelSize[1] + 10)
                    cv2.rectangle(frame, (x1, label_ymin-labelSize[1]-10), 
                                (x1+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (x1, label_ymin-7), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    object_count += 1

            # Calculate FPS
            t_stop = time.perf_counter()
            frame_rate_calc = float(1/(t_stop - t_start))

            # Update FPS buffer
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            
            # Calculate average FPS
            avg_frame_rate = np.mean(frame_rate_buffer)

            # Display FPS and object count
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Objects: {object_count}', (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow('Detection Results', frame)

        except Exception as e:
            print(f"Error in prediction: {str(e)}")

        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Pause
            cv2.waitKey()
        elif key == ord('p'):  # Save frame
            cv2.imwrite('capture.png', frame)

except KeyboardInterrupt:
    print("\nStopping detection...")
except Exception as e:
    print(f"Error in main loop: {str(e)}")

# Stop threads before cleanup
stop_threads = True
if prediction_thread.is_alive():
    prediction_thread.join(timeout=1.0)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
cap.release()
cv2.destroyAllWindows()
if os.path.exists(temp_file):
    os.remove(temp_file)
print("Done!")