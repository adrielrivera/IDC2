import cv2
import time
import numpy as np
import requests
import json
from dotenv import load_dotenv
import os

# Set the Qt backend for OpenCV on Raspberry Pi
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11/xcb backend for Raspberry Pi
os.environ['DISPLAY'] = ':0'  # Ensure display is set

load_dotenv()
api_key = os.getenv('ROBOFLOW_API')

print("Note: Make sure you have started the Roboflow Inference Server locally")
print("Install it with: pip install roboflow-inference-server")
print("Run it with: rfserver start")

# Initialize camera
print("Setting up camera...")
cap = cv2.VideoCapture(0)  # USB camera index 0

# Set resolution (lower for better performance)
resW, resH = 320, 240  # Reduced for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Create window
cv2.namedWindow("Detection Results", cv2.WINDOW_NORMAL)

# Initialize control and status variables
display_fps_buffer = []  # For display frame rate
detection_fps_buffer = []  # For detection frame rate
fps_avg_len = 30
frame_count = 0  # Add frame counter for skipping
latest_predictions = []  # Store latest predictions to show on skipped frames
last_detection_time = time.perf_counter()  # Track time between detections
avg_detection_fps = 0  # Initialize detection FPS

# Set bounding box colors
bbox_colors = [(0, 255, 0)]  # Using green for now

# Roboflow API endpoint
API_URL = f"https://detect.roboflow.com/idc2/13"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
}
PARAMS = {
    "api_key": api_key,
    "confidence": 40,
    "overlap": 30,
}

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
        
        # Only run prediction every 5th frame
        should_predict = frame_count % 5 == 0
        frame_count += 1

        try:
            if should_predict:
                print("\n" + "="*50)
                print("TIMING MEASUREMENTS:")
                
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
                
                print(f"Starting prediction... (Detection FPS: {avg_detection_fps:.1f})")
                
                # Encode image to base64
                predict_start = time.perf_counter()
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_base64 = img_encoded.tobytes()
                
                # Send to Roboflow API
                response = requests.post(
                    API_URL,
                    params=PARAMS,
                    headers=HEADERS,
                    data=image_base64
                )
                predictions = response.json()
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
                total_time = time.perf_counter() - loop_start
                print(f"TOTAL FRAME TIME: {total_time*1000:.1f}ms")
                print("="*50 + "\n")

            # Display FPS and object count
            cv2.putText(frame, f'FPS: {avg_detection_fps:.1f}', (10, 20), 
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

# Clean up
print(f'Average Detection FPS: {avg_detection_fps:.1f}')
cap.release()
cv2.destroyAllWindows()