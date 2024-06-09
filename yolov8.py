import cv2 as cv
from ultralytics import YOLO

def predict(model, img, conf=0.5):
    results = model(img, conf = conf)
    
    bboxes = []
    for result in results:
        bboxes = result.boxes.xywh.cpu().numpy()
    
    for bbox in bboxes:
        bbox[0]-=bbox[2]/2
        bbox[1]-=bbox[3]/2

    return bboxes, results


def predict_and_detect(model, img, conf=0.5):
    results = model([img], conf = conf)
    for i, r in enumerate(results):
        r.show()


# # Read image
# img = cv.imread('assets/images/test17.jpg')
# img = cv.resize(img, (640, 640))

# # Load a model
# model = YOLO("yolo_archive/models/yolov8/v1/best.pt")

# # # Use the model
# predict_and_detect(model, img)

# bboxes, results = predict(model, img)
# print(bboxes)