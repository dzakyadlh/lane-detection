import cv2 as cv

def predict(img, model, conf_th, roi=[0,0,0,0], show=False):
    # Set parameters
    NMS_th = .25

    # Set Region of Interest before detection if needed
    if roi != [0,0,0,0]:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]

    # Run detection
    labels, scores, bboxes = model.detect(img, conf_th, NMS_th)

    # Show bounding box if needed
    if show == True:
        img_yolo = img.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv.rectangle(img_yolo, (x, y), (x+w, y+h), (255, 0, 255), 1)
        return img_yolo, labels, scores, bboxes

    return labels, scores, bboxes

