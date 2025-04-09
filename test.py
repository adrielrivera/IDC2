import cv2
skibidi = cv2.VideoCapture(0)
while True:
    ret, toilet = skibidi.read()
    cv2.imshow('Camera Feed', toilet)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

skibidi.release()
cv2.destroyAllWindows()