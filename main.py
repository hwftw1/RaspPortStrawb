from os.path import isfile, join
from os import listdir
import os
import torch
from PIL import Image
import cv2

#1008 756
#yolov5s
#scaling factor of 2.7

# def path_reader(pathway):
#     paths = [f for f in listdir(pathway) if isfile(join(pathway, f))]
#     out_paths = []
#     for ps in range(0, len(paths)):
#         out_paths.append(paths[ps])
#     return out_paths

def get_pictures():
    cam = cv2.VideoCapture(0)
    cam.set(3, 960)
    cam.set(4, 720)
    ret, frame = cam.read()
    cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('/home/hwickens/code/left.jpg', frame)
    cam.release()
    os.system("libcamera-still --rotation 0 -t 1 -o /home/hwickens/code/right.jpg")


def get_strawbs(img_input):
    img_batch = []
    img = cv2.imread(img_input)
    img = cv2.resize(img, (720, 720))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_batch.append(img)
    results = model(img_batch, size=720)
    results.save()

print("-----LOADING MODEL-----")
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')  # local repo
model.conf = 0.8
if os.path.exists("runs/detect/exp/image0.jpg"):
    os.remove("runs/detect/exp/image0.jpg")
if os.path.exists("runs/detect/exp2/image0.jpg"):
    os.remove("runs/detect/exp2/image0.jpg")
if os.path.exists("runs/detect/exp/"):
    os.rmdir("runs/detect/exp/")
if os.path.exists("runs/detect/exp2/"):
    os.rmdir("runs/detect/exp2/")
print("-----TAKING PICTURES-----")
get_pictures()
print("-----IDENTIFYING STRAWBERRIES (LEFT)-----")
get_strawbs("left.jpg")
print("-----IDENTIFYING STRAWBERRIES (RIGHT)-----")
get_strawbs("right.jpg")