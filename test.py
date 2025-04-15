import cv2

# Initialize with DirectShow backend for better performance on Windows
skibidi = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set lower resolution for better performance
skibidi.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
skibidi.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Print current FPS settings
print(f"Default FPS: {skibidi.get(cv2.CAP_PROP_FPS)}")
print(f"Current frame width: {skibidi.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Current frame height: {skibidi.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

while True:
    ret, toilet = skibidi.read()
    cv2.imshow('Camera Feed', toilet)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

skibidi.release()
cv2.destroyAllWindows()