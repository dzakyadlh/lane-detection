import cv2 as cv

def predict(img, model_path, config_path, conf_th, roi=[0,0,0,0], show=False):
    # Set parameters
    model_file = model_path
    config_file = config_path
    NMS_th = .25


    # Read network model
    net = cv.dnn.readNetFromDarknet(config_file, model_file)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    model = cv.dnn.DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True, crop=False)

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
        cv.imshow('img_yolo',img_yolo)
        cv.waitKey(0)

    return labels, scores, bboxes

