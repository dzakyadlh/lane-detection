import numpy as np
import cv2 as cv
import matplotlib as plt

# Function for thresholding the image
def thresholding(img, h_min, s_min, v_min, h_max, s_max, v_max):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask

# Function to warp image into a bird-eye view
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

# Function to calculate pixel summation on left and right sides of the robot
def calculate_pixel_sum(image):
    # Divide the image vertically into left and right regions
    height, width = image.shape[:2]
    mid_point = width // 2
    left_region = image[:, :mid_point]
    right_region = image[:, mid_point:]

    # Calculate pixel summation in each region
    left_sum = np.sum(left_region)
    right_sum = np.sum(right_region)

    return left_sum, right_sum

def get_histogram(image, min_percentage = 0.1, display = False, region = 1):
    
    if region == 1:
        hist_val = np.sum(image, axis=0)
    else:
        hist_val = np.sum(image[image.shape[0]//region:,:], axis=0)

    max_val = np.max(hist_val)
    min_val = min_percentage*max_val

    index_array = np.where(hist_val >= min_val)
    base_point = int(np.average(index_array))
    print(base_point)
    
    if display:
        img_hist = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for i, intensity in enumerate(hist_val):
            cv.line(img_hist, (i, image.shape[0]), (i, image.shape[0]-intensity//255), (255, 0, 255), 1)
            cv.circle(img_hist, (base_point, image.shape[0]), 20, (0, 0, 255), cv.FILLED)
        return base_point, img_hist
    return base_point


def empty(a):
    pass