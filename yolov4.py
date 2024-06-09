import cv2 as cv

def predict(img, model_path, config_path, conf_th, roi=[0,0,0,0], show=False):
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

    # Set Region of Interest before detection
    if roi != [0,0,0,0]:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]

    # Run detection
    labels, scores, bboxes = model.detect(img, conf_th, NMS_th)

    if show == True:
        for bbox in bboxes:
            x, y, w, h = bbox
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
            cv.imshow('img',img)
            cv.waitKey(0)

    return labels, scores, bboxes
