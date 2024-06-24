import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils_rev
from ultralytics import YOLO
import yolov4
import yolov8

start_time = time.time()

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
# centers, img = utils_rev.draw_centers(img, bboxes)
left_centers, right_centers, img = utils_rev.obtain_centers(img, bboxes)
print(left_centers)
print(right_centers)

slopes, averaged_line, img = utils_rev.hough_transform2(img, left_centers, right_centers, 10, 10, 100, 60, 120, show=True)

# # Run hough transform
# slopes, averaged_line, img = utils_rev.hough_transform2(img, 10, 10, 50, 60, 120, show=True)
# print(slopes)
# print(averaged_line)

print('Runtime(s): ', round((time.time() - start_time),3))
print('FPS: ', round(1/(time.time() - start_time),3))

# cv.imshow('result', img)
# cv.waitKey(0)
plt.imshow(img)
plt.show()