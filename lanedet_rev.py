import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils_rev
from ultralytics import YOLO
import yolov4
import yolov8

# start_time = time.time()

# Read image
img = cv.imread('assets/images/test15.jpg')
img = cv.resize(img, (416, 416))

# Run detection with yolov4
model_file = 'yolo_archive/models/yolov4/v4/yolov4-obj_best.weights'
config_file = 'yolo_archive/yolov4-obj.cfg'
labels, scores, bboxes = yolov4.predict(img, model_file, config_file, 0.5)

# # Run detection with yolov8
# model = YOLO('yolo_archive/models/yolov8/v1/best.pt')
# bboxes, results = yolov8.predict(model, img)

# Draw centers
# left_line, right_line, img = utils_rev.process_bboxes(img, bboxes,threshold=60)
centers, img = utils_rev.draw_centers(img, bboxes)

# Run hough transform
slopes, averaged_line, img = utils_rev.hough_transform(img, 10, 10, 50, 70, 110, show=True)

# Tractor Guidance
dl, dr, dm, guide, img = utils_rev.tractor_guidance(img, averaged_line, 20, show=True)

# print('Runtime(s): ', round((time.time() - start_time),3))
