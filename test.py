import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils_rev
import yolov4

# Read images

img = cv.imread('assets/images/test13.jpg')
img = cv.resize(img, (416, 416))

# Run detection with yolov4
img_yolo = img.copy()
model_file = 'yolo_archive/models/yolov4/v4/yolov4-obj_best.weights'
config_file = 'yolo_archive/yolov4-obj.cfg'
labels, scores, bboxes = yolov4.predict(img_yolo, model_file, config_file, 0.5)
for label, score, bbox in zip(labels, scores, bboxes):
    x, y, w, h = bbox
    right = round(x + w)
    left = round(x)
    bottom = round(y + h)
    top = round(y)
    cv.rectangle(img_yolo, (left, top), (right, bottom), (255, 0, 255), 1)
    cv.putText(img_yolo, "{} [{:.2f}]".format('paddy', round(score, 3)),
                (left, top - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 255), 2)

# Draw centers
img_centers = img.copy()
centers, img_centers = utils_rev.draw_centers(img_centers, bboxes)

# Run Thresholding
img_thresh = utils_rev.thresholding(img_centers.copy(), 150, 150, 255, 255, 255, 255)

img_edges = cv.Canny(img_thresh.copy(), 150, 255, apertureSize=3)

# Run hough transform
img_hough = img_centers.copy()
slopes, averaged_line, img_hough = utils_rev.hough_transform(img_hough, 10, 10, 50, 70, 110, show=True)

titles = ['Original', 'ROI Extraction', 'Paddy Detection', 'Centers', 'Thresholding', 'Canny', 'Hough Transform']
images = [img, img, img_yolo, img_centers, img_thresh, img_edges, img_hough]

for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()