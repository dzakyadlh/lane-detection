import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils

# Read images

img = cv.imread('assets/images/test1.jpg')
img = cv.resize(img, (416, 416))

# Set parameters
model_file = 'yolo_archive/models/yolov4/v3/yolov4-obj_5000.weights'
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

# Draw centers
centers, img_centers = utils.draw_centers(img, bboxes)

# Run Thresholding
img_thresh = utils.thresholding(img_centers, 150, 150, 255, 255, 255, 255)

# Run hough transform
hough_lines = []
lines = cv.HoughLinesP(img_thresh, 1, np.pi/180, 10, minLineLength=5, maxLineGap=100)
if lines is not None:
    lines = np.squeeze(lines)  # Convert lines to a 2D array

    # Making sure the lines go through y1 = 0 to y2 = max y
    lines[:, 1] = 0
    lines[:, 3] = img.shape[0]

    for line in lines:
        img_hough_1 = cv.line(img_centers.copy(), (line[0], line[1]), (line[2], line[3]), color, 2)

    # Making sure the lines are vertical
    angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]) * 180.0 / np.pi
    mask = (angles >= 75) & (angles <= 105)
    lines = lines[mask]

    for line in lines:
        img_hough_2 = cv.line(img_centers.copy(), (line[0], line[1]), (line[2], line[3]), color, 2)

    if len(lines) > 0:
        # Sorting lines based on x1
        lines = lines[np.argsort(lines[:, 0])]

        # Averaging the lines
        indices = np.where(np.abs(np.diff(lines[:, 0])) >= img.shape[1]/7)[0]
        starts = np.concatenate([[0], indices + 1])
        ends = np.concatenate([indices + 1, [len(lines)]])
        for start, end in zip(starts, ends):
            avg_line = np.mean(lines[start:end], axis=0).astype(int)
            hough_lines.append(avg_line)
            img_hough_3 = cv.line(img_centers.copy(), (avg_line[0], avg_line[1]), (avg_line[2], avg_line[3]), color, 2)

titles = ['Original', 'ROI Extraction', 'Paddy Detection', 'Centers', 'Thresholding', 'Hough Transform', 'Removed Horizontal', 'Averaged Hough']
images = [img, img, img_yolo, img_centers, img_thresh, img_hough_1, img_hough_2, img_hough_3]

for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()