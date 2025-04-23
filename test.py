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
print("Press Ctrl+C to stop")

try:
    while True:
        # Capture frame
        ret, image = cam.read()
        
        if ret:
            # Only process every 5th frame for better performance
            if frame_count % 5 == 0:
                # Save frame temporarily for Roboflow
                cv2.imwrite(temp_file, image)
                
                try:
                    # Run prediction on saved frame
                    predictions = model.predict(temp_file, confidence=40, overlap=30).json()
                    
                    # Clear the console and print detection info
                    os.system('clear')  # 'cls' on Windows
                    
                    # Calculate and display FPS
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    
                    print(f"FPS: {int(fps)}")
                    print(f"Frame size: {image.shape[1]}x{image.shape[0]}")
                    print("\nDetections:")
                    
                    if len(predictions['predictions']) == 0:
                        print("No objects detected")
                    
                    # Print detection results
                    for prediction in predictions['predictions']:
                        # Get coordinates
                        x1 = prediction['x'] - prediction['width'] / 2
                        y1 = prediction['y'] - prediction['height'] / 2
                        x2 = prediction['x'] + prediction['width'] / 2
                        y2 = prediction['y'] + prediction['height'] / 2
                        
                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Print detection info
                        print(f"* {prediction['class']} (confidence: {prediction['confidence']:.2f})")
                        print(f"  Position: ({x1},{y1}) to ({x2},{y2})")
                        
                except Exception as e:
                    print(f"Error during prediction: {str(e)}")
            
            # Increment frame counter
            frame_count += 1
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping detection...")

# Clean up
print("Cleaning up...")
cam.release()
if os.path.exists(temp_file):
    os.remove(temp_file)
print("Done!")