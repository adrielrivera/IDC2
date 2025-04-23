import cv2
import time
from roboflow import Roboflow
import os

# Initialize Roboflow model
rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")
project = rf.workspace().project("idc2")
model = project.version("10").model

# Initialize camera
skibidi = cv2.VideoCapture(0)

# lowres baby
skibidi.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
skibidi.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# For FPS calculation
prev_time = time.time()
fps = 0

# Temp file for saving frames
temp_file = "temp_frame.jpg"

while True:
    # Capture frame
    ret, toilet = skibidi.read()
    
    if ret:
        # Save frame temporarily for Roboflow
        cv2.imwrite(temp_file, toilet)
        
        # Get predictions from Roboflow
        try:
            # Use infer on image instead of predict_video for real-time
            predictions = model.predict(temp_file, confidence=40, overlap=30).json()
            
            # Draw predictions on frame
            for prediction in predictions['predictions']:
                # Get coordinates
                x1 = prediction['x'] - prediction['width'] / 2
                y1 = prediction['y'] - prediction['height'] / 2
                x2 = prediction['x'] + prediction['width'] / 2
                y2 = prediction['y'] + prediction['height'] / 2
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(toilet, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{prediction['class']} {prediction['confidence']:.2f}"
                cv2.putText(toilet, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            # Show error on frame
            cv2.putText(toilet, f"Error: {str(e)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS
        cv2.putText(toilet, f"FPS: {int(fps)}", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Food Plushie Detector', toilet)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
skibidi.release()
cv2.destroyAllWindows()
if os.path.exists(temp_file):
    os.remove(temp_file)