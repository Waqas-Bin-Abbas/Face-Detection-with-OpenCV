import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.imread('test1.jpg')

gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                     minNeighbors=5, minSize=(25, 25))

for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('image', gray)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
