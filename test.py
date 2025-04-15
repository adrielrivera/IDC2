import cv2
skibidi = cv2.VideoCapture(0)

# Try to set maximum FPS (common values are 30, 60, or 120)
skibidi.set(cv2.CAP_PROP_FPS, 60)  # Try 60 FPS first

# Print current FPS settings
print(f"Current FPS: {skibidi.get(cv2.CAP_PROP_FPS)}")
print(f"Current frame width: {skibidi.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Current frame height: {skibidi.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

while True:
    ret, toilet = skibidi.read()
    cv2.imshow('Camera Feed', toilet)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

skibidi.release()
cv2.destroyAllWindows()