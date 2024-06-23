import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import utils_rev
from ultralytics import YOLO
import torch


def predict(model, img, conf=0.5):
    results = model(img)
    print(results.xywh[0])

    bboxes = []
    for result in results.xywh[0]:
        bboxes = result.boxes.xywh.cpu().numpy()
    
    for bbox in bboxes:
        bbox[0]-=bbox[2]/2
        bbox[1]-=bbox[3]/2

    return bboxes, results

# Function to process a single frame
def process_frame(frame, model):
    # Resize frame
    frame_resized = cv.resize(frame, (416, 416))

    # Run detection with yolov5
    bboxes, results = predict(model, frame)

    # Draw centers
    centers, frame_centers = utils_rev.draw_centers(frame_resized, bboxes)

    # # Run hough transform
    # slopes, averaged_line, frame_hough = utils_rev.hough_transform(frame_centers, 10, 10, 50, 70, 110, show=True)

    return frame_final

# Take input
cap = cv.VideoCapture('assets/videos/finaltest.mp4')
model = torch.hub.load(r'yolov5', 'custom', path=r'yolo_archive/models/yolov5/nano/best.pt', source='local')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        start_time = time.time()

        # Process frame
        frame_final = process_frame(frame, model)

        # Display the resulting frame
        cv.imshow('frame', frame_final)

        # Print FPS
        print("FPS: ", 1.0 / (time.time() - start_time))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
cv.destroyAllWindows()