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