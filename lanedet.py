import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils

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

img = cv.imread('assets/images/test1.jpg')
img = cv.resize(img, (416, 416))

# Set Region of Interest before detection
# img = img[y:y+h, x:x+w]

# Run detection
labels, scores, bboxes = model.detect(img, conf_th, NMS_th)
print(labels)
print(scores)
print(bboxes)
# for (labelid, score, box) in zip(labels, scores, bboxes):
    # cv.rectangle(img, box, color, 1)

# Draw centers
centers, img = utils.draw_centers(img, bboxes)

# Run hough transform
# img = utils.hough_transform(img, 10, 80, 100, 60)
lines, img = utils.probabilistic_hough_transform(img, 10, 5, 100, 75, 105, img.shape[1]/7)
print(lines)

# Or Linearization
# lines, img = utils.lines_linearization(img, centers, 60)

# # Calculate angle of Hough line
# angle_left, angle_right = utils.calculate_angle(lines)
# img = cv.putText(img, str(angle_left), (50,400), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
# img = cv.putText(img, str(angle_right), (300,400), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Tractor guidance
guidance = utils.tractor_guidance(img, lines)
print(guidance)

print('Runtime(s): ', round((time.time() - start_time),3))

cv.imshow('img', img)
cv.waitKey(0)
# plt.imshow(img)
# plt.show()

# darknet.exe detector test data/obj.data cfg/yolov4-obj.cfg weights/yolov4-obj_best.weights -ext_output

# # Take input
# cap = cv.VideoCapture('assets/videos/rice_ss.mp4')

# # cap.set(3, 800)
# # cap.set(4, 600)

# # recorded = cv2.VideoWriter('recorded_vid.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # recorded.write(frame)

#         frame = cv.resize(frame, (800, 600))
#         xmax, ymax = frame.shape[1], frame.shape[0]
#         # Run detection
#         labels, scores, bboxes = model.detect(frame, conf_th, NMS_th)
#         # for (labelid, score, box) in zip(labels, scores, bboxes):
#             # cv.rectangle(frame, box, color, 1)

#         # Define ROI on the image and eliminate outliers
#         roi1, roi2 = round(0.1*xmax), round(0.9*xmax)

#         # Draw bounding box centers
#         centers, frame = utils.draw_centers(frame, bboxes, roi1, roi2, color)

#         # Run hough transform
#         frame = utils.hough_transform(frame, 10, 80, 100, frame.shape[1]/7)
#         lines, frame = utils.probabilistic_hough_transform(frame, 10, 5, 100, 75, 105, frame.shape[1]/7)

#         # Or Linearization
#         # lines, frame = utils.lines_linearization(frame, centers, frame.shape[1]/7)

#         # Calculate angle of Hough line
#         # angle_left, angle_right = utils.calculate_angle(lines)
#         # frame = cv.putText(frame, str(angle_left), (50,400), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         # frame = cv.putText(frame, str(angle_right), (300,400), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#         cv.imshow('frame', frame)

#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# # recorded.release()
# cv.destroyAllWindows()