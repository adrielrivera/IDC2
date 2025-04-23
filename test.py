import cv2
import time
from roboflow import Roboflow
import os

# Initialize Roboflow model
print("Initializing Roboflow model...")
rf = Roboflow(api_key="q4Y1pRJA0SETfWqL4kKU")
project = rf.workspace().project("idc2")
model = project.version("10").model
print("Model initialized!")

# Initialize camera
print("Setting up camera...")
cam = cv2.VideoCapture(0)

# Set resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera ready!")

# Temp file for saving frames
temp_file = "temp_frame.jpg"

# For FPS calculation
prev_time = time.time()
frame_count = 0

print("Starting detection loop...")
while True:
    # Capture frame
    ret, image = cam.read()
    
    if ret:
        # Process every 5th frame for better performance
        if frame_count % 5 == 0:
            # Save frame temporarily for Roboflow
            cv2.imwrite(temp_file, image)
            
            try:
                # Run prediction on saved frame
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
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{prediction['class']} {prediction['confidence']:.2f}"
                    cv2.putText(image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Camera Feed', image)
        
        # Increment frame counter
        frame_count += 1
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("Cleaning up...")
cam.release()
cv2.destroyAllWindows()
if os.path.exists(temp_file):
    os.remove(temp_file)
print("Done!")