import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils_rev

start_time = time.time()

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
# print(class_name)


# Read network model
net = cv.dnn.readNetFromDarknet(config_file, model_file)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)


model = cv.dnn.DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True, crop=False)

img = cv.imread('assets/images/test10.jpg')
img = cv.resize(img, (416, 416))

# Set Region of Interest before detection
# img = img[y:y+h, x:x+w]

# Run detection
labels, scores, bboxes = model.detect(img, conf_th, NMS_th)
# for bbox in bboxes:
#     cv.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 255), 1)
# cv.imshow('img',img)
# cv.waitKey(0)

# Draw centers
left_line, right_line, img = utils_rev.process_bboxes(img, bboxes,threshold=60)

# Run hough transform
# lines, img = utils_rev.probabilistic_hough_transform(img, 10, 10, 30, 60, 120, 30)
slopes, averaged_line, img = utils_rev.probabilistic_hough_transform2(img, 10, 10, 50, 80, 110, 30)

# Or Linearization
# left_line, right_line, img = utils_rev.lines_linearization(img, left_line, right_line)

# Tractor Guidance
ml, mr, dm, img = utils_rev.tractor_guidance2(img, slopes)

print('Runtime(s): ', round((time.time() - start_time),3))

cv.imshow('img', img)
cv.waitKey(0)
# plt.imshow(img)
# plt.show()
