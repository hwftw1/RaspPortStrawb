import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 960)
cam.set(4, 720)
ret, frame = cam.read()
cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('/home/hwickens/code/left.jpg', frame)
cam.release()
os.system("libcamera-still --rotation 0 -t 1 -o /home/hwickens/code/right.jpg")