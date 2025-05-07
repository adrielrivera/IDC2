import cv2
import time
from roboflow import Roboflow
import os
import threading
import queue
import numpy as np
from dotenv import load_dotenv

# Set the Qt backend for OpenCV on Raspberry Pi
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11/xcb backend for Raspberry Pi
os.environ['DISPLAY'] = ':0'  # Ensure display is set

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
resW, resH = 640, 480  # Changed from 1920x1080 to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Set buffer size to minimum
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera ready!")

# Temp file for saving frames
temp_file = "temp_frame.jpg"

# Initialize control and status variables
display_fps_buffer = []  # For display frame rate
detection_fps_buffer = []  # For detection frame rate
fps_avg_len = 30
frame_count = 0  # Add frame counter for skipping
latest_predictions = []  # Store latest predictions to show on skipped frames
last_detection_time = time.perf_counter()  # Track time between detections
avg_detection_fps = 0  # Initialize detection FPS

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
        loop_start = time.perf_counter()

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print('Unable to read frames from the camera. Camera may be disconnected. Exiting program.')
            break
        
        # Only run prediction every 3rd frame
        should_predict = frame_count % 3 == 0
        frame_count += 1

        # Run prediction on frame
        try:
            if should_predict:
                # Calculate detection FPS
                current_time = time.perf_counter()
                detection_interval = current_time - last_detection_time
                detection_fps = 1.0 / detection_interval if detection_interval > 0 else 0
                last_detection_time = current_time
                
                # Update detection FPS buffer
                if len(detection_fps_buffer) >= fps_avg_len:
                    detection_fps_buffer.pop(0)
                detection_fps_buffer.append(detection_fps)
                avg_detection_fps = np.mean(detection_fps_buffer) if detection_fps_buffer else 0
                
                print("\n" + "="*50)
                print("TIMING MEASUREMENTS:")
                print(f"Starting prediction... (Detection FPS: {avg_detection_fps:.1f})")
                
                predict_start = time.perf_counter()
                predictions = model.predict(frame, confidence=40, overlap=30).json()
                predict_time = time.perf_counter() - predict_start
                print(f"Prediction completed: {predict_time*1000:.1f}ms")
                latest_predictions = predictions

            # Initialize variable for basic object counting
            object_count = 0

            # Draw predictions
            draw_start = time.perf_counter()
            if 'predictions' in latest_predictions:
                for prediction in latest_predictions['predictions']:
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
            draw_time = time.perf_counter() - draw_start

            if should_predict:
                print(f"Drawing time: {draw_time*1000:.1f}ms")
                # Calculate display FPS
                display_time = time.perf_counter() - loop_start
                display_fps = 1.0 / display_time if display_time > 0 else 0
                
                # Update display FPS buffer
                if len(display_fps_buffer) >= fps_avg_len:
                    display_fps_buffer.pop(0)
                display_fps_buffer.append(display_fps)
                avg_display_fps = np.mean(display_fps_buffer)

                # Calculate total time for this frame
                total_time = time.perf_counter() - loop_start
                print(f"TOTAL FRAME TIME: {total_time*1000:.1f}ms")
                print(f"Current FPS: {1/total_time:.2f}")
                print("="*50 + "\n")

            # Display FPS and object count
            cv2.putText(frame, f'Display FPS: {avg_display_fps:.1f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Detection FPS: {avg_detection_fps:.1f}', (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Objects: {object_count}', (10, 80),
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
print(f'Average pipeline FPS: {avg_display_fps:.2f}')
cap.release()
cv2.destroyAllWindows()
if os.path.exists(temp_file):
    os.remove(temp_file)
print("Done!")