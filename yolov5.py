import numpy as np
from ultralytics import YOLO


def show(model, img):
    results = model(img)
    results.show()

def predict(model, img):
    results = model(img)

    # Extract bounding boxes directly
    bboxes = np.round(results.xywh[0].cpu().numpy()).astype(int)

    return bboxes