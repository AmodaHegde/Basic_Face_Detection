import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('D:\FaceDetection\haarcascade_frontalface_default.xml')
image = cv2.imread('D:\FaceDetection\pic2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
