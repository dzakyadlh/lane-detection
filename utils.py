import numpy as np
import cv2 as cv
import matplotlib as plt
import seaborn as sns
import moviepy as mpy
import scipy

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

def empty(a):
    pass