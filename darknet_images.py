import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet

def check_arguments_errors(thresh, config_file, weights, data_file, input):
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(config_file))))
    if not os.path.exists(weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(weights))))
    if not os.path.exists(data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(data_file))))
    if input and not os.path.exists(input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    return batch_array

def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    batch_array = prepare_batch(images, network)
    batch_array = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = darknet.IMAGE(image_width, image_height, 3, batch_array)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def extract_coordinates(image, detections):
    bbox_coordinates = []
    for label, confidence, bbox in detections:
        x, y, w, h = convert2relative(image, bbox)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        bbox_coordinates.append(np.array([x1, y1, x2, y2]))
    bbox_array = np.array(bbox_coordinates)
    return bbox_array


def main(input="", batch_size=1, config_file="./cfg/yolov4-obj.cfg", data_file="./data/obj.data", weights="./weights/yolov4-obj_best.weights", thresh=.25, dont_show=False, ext_output=False, save_labels=False):
    check_arguments_errors(thresh, config_file, weights, data_file, input)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )

    images = load_images(input)

    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, thresh
            )
        if save_labels:
            save_annotations(image_name, image, detections, class_names)
        darknet.print_detections(detections, ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        extract_coordinates(image_name, detections)
        if not dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        index += 1

if __name__ == "__main__":
    main()

# network_width
# network_height
# make_image
# copy_image_from_bytes
# detect_image
# free_image
# draw_boxes
# POINTER
# IMAGE
# do_nms_obj
# remove_negatives
# free_batch_detections
# load_network
# print_detections