import cv2 as cv
import numpy as np


def empty(a):
    pass


cv.namedWindow('HSV')
cv.resizeWindow('HSV', 640, 240)
cv.createTrackbar('Hue Min', 'HSV', 0, 179, empty)
cv.createTrackbar('Hue Max', 'HSV', 179, 179, empty)
cv.createTrackbar('Sat Min', 'HSV', 0, 255, empty)
cv.createTrackbar('Sat Max', 'HSV', 255, 255, empty)
cv.createTrackbar('Val Min', 'HSV', 0, 255, empty)
cv.createTrackbar('Val Max', 'HSV', 255, 255, empty)

while True:
    img = cv.imread('./assets/images/sawah1.jpg')
    imgHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos('Hue Min', 'HSV')
    h_max = cv.getTrackbarPos('Hue Max', 'HSV')
    s_min = cv.getTrackbarPos('Sat Min', 'HSV')
    s_max = cv.getTrackbarPos('Sat Max', 'HSV')
    v_min = cv.getTrackbarPos('Val Min', 'HSV')
    v_max = cv.getTrackbarPos('Val Max', 'HSV')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHsv, lower, upper)
    result = cv.bitwise_and(img, img, mask=mask)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])

    cv.imshow("Result", hStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
