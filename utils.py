import numpy as np
import cv2 as cv
import matplotlib as plt
import seaborn as sns
import moviepy as mpy
import scipy

def thresholding(img, h_min, s_min, v_min, h_max, s_max, v_max):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask

def warp_img(img, points, width, height):
    pt1 = np.float32(points)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(pt1, pt2)
    warped = cv.warpPerspective(img, matrix, (width, height))
    return warped

def initialize_points_trackbars(initial, w=480, h=240):
    cv.namedWindow('Points Trackbars')
    cv.resizeWindow('Points Trackbars', 360, 240)
    cv.createTrackbar('Width Top', 'Points Trackbars', initial[0], w/2, empty)
    cv.createTrackbar('Height Top', 'Points Trackbars', initial[1], h, empty)
    cv.createTrackbar('Width Bottom', 'Points Trackbars', initial[2], w/2, empty)
    cv.createTrackbar('Height Bottom', 'Points Trackbars', initial[3], h, empty)

def get_trackbar_points(w=480, h=240):
    width_top = cv.getTrackbarPos('Width Top', 'Points Trackbars')
    height_top = cv.getTrackbarPos('Height Top', 'Points Trackbars')
    width_bottom = cv.getTrackbarPos('Width Bottom', 'Points Trackbars')
    height_bottom = cv.getTrackbarPos('Height Bottom', 'Points Trackbars')
    points = np.float32([(width_top, height_top), (w-width_top, height_top), (width_bottom, height_bottom), (w-width_bottom, height_bottom)])
    return points

def empty(a):
    pass