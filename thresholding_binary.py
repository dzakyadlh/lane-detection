import cv2 as cv
import numpy as np


def empty(a):
    pass

def initialize_trackbar():
    cv.namedWindow('Th')
    cv.resizeWindow('Th', 640, 240)
    cv.createTrackbar('Lower', 'Th', 0, 255, empty)
    cv.createTrackbar('Upper', 'Th', 0, 255, empty)

def get_trackbar_positions():
    lower = cv.getTrackbarPos('Lower', 'Th')
    upper = cv.getTrackbarPos('Upper', 'Th')

    return lower, upper

def detect_color_img(img):
    initialize_trackbar()

    #by image
    while True:
        lower, upper = get_trackbar_positions()

        _, img_th = cv.threshold(img, lower, upper, cv.THRESH_BINARY)

        h_stack = np.hstack([img, img_th])

        cv.imshow("Result", h_stack)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

img = cv.imread('test2.jpg')
img = cv.resize(img, (512, 512))
detect_color_img(img)