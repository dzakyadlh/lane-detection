# def batch_detection_example():
#     args = parser()
#     check_arguments_errors(args)
#     batch_size = 3
#     random.seed(3)  # deterministic bbox colors
#     network, class_names, class_colors = darknet.load_network(
#         args.config_file,
#         args.data_file,
#         args.weights,
#         batch_size=batch_size
#     )
#     image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
#     images = [cv2.imread(image) for image in image_names]
#     images, detections,  = batch_detection(network, images, class_names,
#                                            class_colors, batch_size=batch_size)
#     for name, image in zip(image_names, images):
#         cv2.imwrite(name.replace("data/", ""), image)
#     print(detections)


# calling 'performDetect' function from darknet.py
from darknet-master.darknet import performDetect as scan
import math


def detect(img_path):
    ''' this script if you want only want get the coord '''
    picpath = img_path
    # change this if you want use different config
    cfg = 'darknet-master/cfg/yolov3-obj.cfg'
    coco = 'darknet-master/data/obj.data'  # you can change this too
    # and this, can be change by you
    data = 'darknet-master/data/backup/yolov3-obj_5000.weights'
    test = scan(imagePath=picpath, thresh=0.25, configPath=cfg, weightPath=data, metaPath=coco, showImage=False, makeImageOnly=False,
                initOnly=False)  # default format, i prefer only call the result not to produce image to get more performance

    # until here you will get some data in default mode from alexeyAB, as explain in module.
    # try to: help(scan), explain about the result format of process is: [(item_name, convidence_rate (x_center_image, y_center_image, width_size_box, height_size_of_box))],
    # to change it with generally used form, like PIL/opencv, do like this below (still in detect function that we create):

    newdata = []

    # For multiple Detection
    if len(test) >= 2:
        for x in test:
            item, confidence_rate, imagedata = x
            x1, y1, w_size, h_size = imagedata
            x_start = round(x1 - (w_size/2))
            y_start = round(y1 - (h_size/2))
            x_end = round(x_start + w_size)
            y_end = round(y_start + h_size)
            data = (item, confidence_rate,
                    (x_start, y_start, x_end, y_end), (w_size, h_size))
            newdata.append(data)

    # For Single Detection
    elif len(test) == 1:
        item, confidence_rate, imagedata = test[0]
        x1, y1, w_size, h_size = imagedata
        x_start = round(x1 - (w_size/2))
        y_start = round(y1 - (h_size/2))
        x_end = round(x_start + w_size)
        y_end = round(y_start + h_size)
        data = (item, confidence_rate,
                (x_start, y_start, x_end, y_end), (w_size, h_size))
        newdata.append(data)

    else:
        newdata = False

    return newdata


if __name__ == "__main__":
    # Multiple detection image test
    # table = '/home/saggi/Documents/saggi/prabin/darknet/data/26.jpg'
    # Single detection image test
    table = 'test2.jpg'
    detections = detect(table)
