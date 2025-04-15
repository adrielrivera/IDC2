import cv2


skibidi = cv2.VideoCapture(0)


# lowres baby
skibidi.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
skibidi.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



while True:
    ret, toilet = skibidi.read()
    cv2.imshow('Camera Feed', toilet)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

skibidi.release()
cv2.destroyAllWindows()