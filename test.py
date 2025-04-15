import cv2
import time

# Initialize with DirectShow backend for better performance on Windows
skibidi = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Optimize camera settings
skibidi.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
skibidi.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
skibidi.set(cv2.CAP_PROP_FPS, 30)
skibidi.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size

# Print current settings
print(f"Default FPS: {skibidi.get(cv2.CAP_PROP_FPS)}")
print(f"Current frame width: {skibidi.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Current frame height: {skibidi.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Variables for FPS calculation
prev_frame_time = time.time()
curr_frame_time = 0
frame_count = 0
start_time = time.time()

while True:
    # Skip frames if we're falling behind
    if frame_count % 2 == 0:  # Process every other frame if needed
        ret, toilet = skibidi.read()
        if not ret:
            break
        
        # Calculate FPS
        curr_frame_time = time.time()
        frame_count += 1
        if curr_frame_time - start_time >= 1.0:  # Update FPS every second
            fps = frame_count / (curr_frame_time - start_time)
            frame_count = 0
            start_time = curr_frame_time
        
        # Display FPS on frame
        cv2.putText(toilet, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Camera Feed', toilet)
    
    # Use waitKey(1) for minimum delay
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

skibidi.release()
cv2.destroyAllWindows()