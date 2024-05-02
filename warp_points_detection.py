import numpy as np
import cv2 as cv
import utils

def warp_img(img, points, width, height):
    pt1 = np.float32(points)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pt1, pt2)
    warped = cv.warpPerspective(img, matrix, (width, height))
    return warped

def initialize_points_trackbars(initial, w=480, h=240):
    cv.namedWindow('Points Trackbars')
    cv.resizeWindow('Points Trackbars', 360, 240)
    cv.createTrackbar('Width Top', 'Points Trackbars', initial[0], w//2, empty)
    cv.createTrackbar('Height Top', 'Points Trackbars', initial[1], h, empty)
    cv.createTrackbar('Width Bottom', 'Points Trackbars', initial[2], w//2, empty)
    cv.createTrackbar('Height Bottom', 'Points Trackbars', initial[3], h, empty)

def get_trackbar_points(w=480, h=240):
    width_top = cv.getTrackbarPos('Width Top', 'Points Trackbars')
    height_top = cv.getTrackbarPos('Height Top', 'Points Trackbars')
    width_bottom = cv.getTrackbarPos('Width Bottom', 'Points Trackbars')
    height_bottom = cv.getTrackbarPos('Height Bottom', 'Points Trackbars')
    points = np.float32([(width_top, height_top), (w-width_top, height_top), (width_bottom, height_bottom), (w-width_bottom, height_bottom)])
    return points

def draw_points(img, points):
    for x in range(4):
        cv.circle(img, (int(points[x][0]), int(points[x][1])), 5, (0, 0, 255), cv.FILLED)
    return img

def empty(a):
    pass

# initial_points = [100, 100, 100, 100]
# initialize_points_trackbars(initial_points)
# while True:
#     lane_img = cv.imread('./assets/images/sawah1.jpg')
#     lane_img = cv.resize(lane_img, (480, 240))
#     lane_img_copy = lane_img.copy()
#     hsv = cv.cvtColor(lane_img, cv.COLOR_BGR2HSV)
#     threshold_img = utils.thresholding(lane_img, 0, 0, 0, 36, 255, 255)
#     h, w, c = lane_img.shape
#     points = get_trackbar_points()
#     warped = warp_img(lane_img, points, w, h)
#     warp_points = draw_points(lane_img_copy, points)

#     cv.imshow('Threshold Image', threshold_img)
#     cv.imshow('Warped Image', warped)
#     cv.imshow('Warp Points', warp_points)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cv.destroyAllWindows()

def detect_warp_points(cap):
    initial_points = [100, 100, 100, 100]
    initialize_points_trackbars(initial_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        lane_img = cv.resize(frame, (480, 240))
        lane_img_copy = lane_img.copy()
        hsv = cv.cvtColor(lane_img, cv.COLOR_BGR2HSV)
        threshold_img = utils.thresholding(lane_img, 0, 0, 62, 36, 54, 110)
        h, w, c = lane_img.shape
        points = get_trackbar_points()
        print(points)
        warped_thresholded = warp_img(threshold_img, points, w, h)
        warped = warp_img(lane_img, points, w, h)
        warp_points = draw_points(lane_img_copy, points)

        cv.imshow('Threshold Image', warped_thresholded)
        cv.imshow('Warped Image', warped)
        cv.imshow('Warp Points', warp_points)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
