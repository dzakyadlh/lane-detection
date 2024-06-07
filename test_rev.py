import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils_rev

# Read images

img = cv.imread('assets/images/test6.jpg')
img = cv.resize(img, (416, 416))

# Set parameters
model_file = 'yolo_archive/models/yolov4/v4/yolov4-obj_best.weights'
config_file = 'yolo_archive/yolov4-obj.cfg'
conf_th = .25
NMS_th = .25
color = (255, 0, 255)

# Read class names
class_name = []
with open('yolo_archive/obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Read network model
net = cv.dnn.readNetFromDarknet(config_file, model_file)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)


model = cv.dnn.DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True, crop=False)

# Run detection
labels, scores, bboxes = model.detect(img, conf_th, NMS_th)
img_yolo = img.copy()
for label, score, bbox in zip(labels, scores, bboxes):
    x, y, w, h = bbox
    right = round(x + w)
    left = round(x)
    bottom = round(y + h)
    top = round(y)
    cv.rectangle(img_yolo, (left, top), (right, bottom), (255, 0, 255), 1)
    cv.putText(img_yolo, "{} [{:.2f}]".format(class_name[label], round(score, 3)),
                (left, top - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 255), 2)

# Draw corners
img_corners = img.copy()
left_line, right_line, img_corners = utils_rev.process_bboxes(img_corners, bboxes, threshold=35)

# Run Thresholding
img_thresh = utils_rev.thresholding(img_corners.copy(), 150, 150, 255, 255, 255, 255)

# Run linearization
img_linear = img.copy()
utils_rev.lines_linearization2(img_linear, left_line, right_line)

titles = ['Original', 'ROI Extraction', 'Paddy Detection', 'Corners', 'Thresholding', 'Linarized']
images = [img, img, img_yolo, img_corners, img_thresh, img_linear]

for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()