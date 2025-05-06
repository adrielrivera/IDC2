import cv2
import time
from roboflow import Roboflow
import os
import threading
import queue

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
rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")
project = rf.workspace().project("idc2")
model = project.version("13").model
print("Model initialized!")

# Initialize camera
print("Setting up camera...")
cam = cv2.VideoCapture(0)

# Set resolution (lower for better performance)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set buffer size to minimum
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera ready!")

# Temp file for saving frames
temp_file = "temp_frame.jpg"

# For FPS calculation
prev_time = time.time()
frame_count = 0
fps_update_interval = 10  # Update FPS every 10 frames
fps = 0

# Thread control
stop_threads = False
frame_queue = queue.Queue(maxsize=1)  # Only store the latest frame

# Start prediction thread
prediction_thread = threading.Thread(target=prediction_worker)
prediction_thread.daemon = True
prediction_thread.start()

print("Starting detection loop...")
print("Press 'q' to quit")

try:
    while True:
        # Capture frame
        ret, image = cam.read()
        
        if ret:
            # Add frame to processing queue every 10th frame
            if frame_count % 10 == 0 and frame_queue.empty():
                # Put a copy of the frame in the queue
                frame_queue.put(image.copy())
            
            # Get predictions from queue if available
            if not prediction_queue.empty():
                latest_predictions = prediction_queue.get()
            
            # Draw predictions on frame
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
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{prediction['class']} {prediction['confidence']:.2f}"
                    cv2.putText(image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            frame_count += 1
            
            if frame_count % fps_update_interval == 0:
                current_time = time.time()
                # Calculate FPS over multiple frames for stability
                fps = fps_update_interval / (current_time - prev_time)
                prev_time = current_time
            
            # Display FPS and frame count
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the resulting frame
            cv2.imshow("Camera Feed", image)
            
            # Break loop with 'q' key - use a shorter wait time for better responsiveness
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping detection...")
except Exception as e:
    print(f"Error in main loop: {str(e)}")

# Stop threads before cleanup
stop_threads = True
if prediction_thread.is_alive():
    prediction_thread.join(timeout=1.0)

# Clean up
print("Cleaning up...")
cam.release()
cv2.destroyAllWindows()
if os.path.exists(temp_file):
    os.remove(temp_file)
print("Done!")