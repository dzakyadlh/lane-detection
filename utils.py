import math
import numpy as np
import cv2 as cv
from sklearn.linear_model import RANSACRegressor

# Draw center points for each bounding boxes
def draw_centers(img, bboxes, color=(255,0,255)):
    centers = []
    for bbox in bboxes:
        x, y, w, h, = bbox
        center_x = round(x + w / 2)
        center_y = round(y + h / 2)
        cv.circle(img, [center_x, center_y], 5, color, -1)
        centers.append([center_x, center_y])
    return centers, img

# Function for thresholding the image
def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask

def hough_transform(img, intersect, min_angle, max_angle, max_xgap, color=(0, 0, 255)):
    hough_lines = []
    img_thresh = thresholding(img, 150, 150, 255, 255, 255, 255)
    edges = cv.Canny(img_thresh, 150, 255, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180,intersect)
    if lines is not None:

        # Extract each coordinates
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            # Making sure the lines are vertical
            angle = np.arctan2(y2-y1, x2-x1)*180.0/np.pi
            if angle >= min_angle and angle <= max_angle:
                hough_lines.append([x1, y1, x2, y2])

    # Averaging the lines
    hough_lines = np.array(sorted(hough_lines, key=lambda x : x[0]))
    n_hough = len(hough_lines)
    if n_hough:
        n = 1
        avg = hough_lines[0]
        for i in range(n_hough-1):
            if abs(hough_lines[i][0]-hough_lines[i+1][0]) < max_xgap:
                avg += hough_lines[i+1]
                n += 1
            else:
                avg //= n
                cv.line(img,(avg[0],avg[1]),(avg[2],avg[3]),color,2)
                avg = hough_lines[i+1]
                n = 1

    return img

def probabilistic_hough_transform(img, intersect, min_line_length, max_line_gap, min_angle, max_angle, max_xgap, color=(0, 0, 255)):
    hough_lines = []
    img_thresh = thresholding(img, 150, 150, 255, 255, 255, 255)  # You need to define `thresholding` function
    edges = cv.Canny(img_thresh, 150, 255, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, intersect, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        lines = np.squeeze(lines)  # Convert lines to a 2D array

        # Making sure the lines go through y1 = 0 to y2 = max y
        lines[:, 1] = 0
        lines[:, 3] = img.shape[0]

        # Making sure the lines are vertical
        angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]) * 180.0 / np.pi
        mask = (angles >= min_angle) & (angles <= max_angle)
        lines = lines[mask]

        if len(lines) > 0:
            # Sorting lines based on x1
            lines = lines[np.argsort(lines[:, 0])]

            # Averaging the lines
            indices = np.where(np.abs(np.diff(lines[:, 0])) >= max_xgap)[0]
            starts = np.concatenate([[0], indices + 1])
            ends = np.concatenate([indices + 1, [len(lines)]])
            for start, end in zip(starts, ends):
                avg_line = np.mean(lines[start:end], axis=0).astype(int)
                hough_lines.append(avg_line)
                cv.line(img, (avg_line[0], avg_line[1]), (avg_line[2], avg_line[3]), color, 2)

    return hough_lines, img

# Linearization method
def lines_linearization(img, centers, max_xgap):
    # Sort the centers array by x-coordinate
    centers = np.array(sorted(centers, key=lambda x: x[0]))

    # Initialize variables needed
    lines = []
    n = len(centers)
    pivot = centers[0]
    highest, lowest = pivot, pivot
    y1, y2 = 0, img.shape[0]  # y coordinates for drawing lines

    # Iterate through all centers
    for i in range(1, n):
        if abs(pivot[0] - centers[i][0]) < max_xgap:  # Check if centers are close in x-direction
            # Update line if the y is lower or higher
            if centers[i][1] < lowest[1]:
                lowest = centers[i]
            elif centers[i][1] > highest[1]:
                highest = centers[i]
        else:
            # Add the line defined by the lowest and highest points in the x-range
            lines.append([lowest[0], y1, highest[0], y2])
            # Update pivot
            pivot = centers[i]
            # Reset highest and lowest coordinates
            highest, lowest = pivot, pivot

     # Add the last line defined by the lowest and highest points in the x-range
    lines.append([lowest[0], y1, highest[0], y2])
    # Draw all the lines on the image
    for line in lines:
        cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

    return lines, img

def ransac_lines_linearization(img, centers):
    # Sort the centers array by x-coordinate
    centers = np.array(sorted(centers, key=lambda x: x[0]))

    # Convert centers to numpy array
    centers = np.array(centers)

    # Initialize variables needed
    lines = []
    y1, y2 = 0, img.shape[0]  # y coordinates for drawing lines

    # Fit lines using RANSAC
    ransac = RANSACRegressor()

    if len(centers) < 2:
        return lines, img

    # Use RANSAC to fit a line model
    ransac.fit(centers[:, 0].reshape(-1, 1), centers[:, 1])
    inlier_mask = ransac.inlier_mask_

    # Extract inliers and outliers
    inliers = centers[inlier_mask]
    outliers = centers[~inlier_mask]

    # Sort inliers by x-coordinate
    inliers = inliers[np.argsort(inliers[:, 0])]

    # Create lines from inliers
    if len(inliers) > 0:
        x1, y1 = inliers[0][0], inliers[0][1]
        x2, y2 = inliers[-1][0], inliers[-1][1]
        lines.append([x1, y1, x2, y2])
        cv.line(img, (x1, 0), (x2, img.shape[0]), (0, 0, 255), 2)

    # Process remaining outliers if any
    while len(outliers) > 0:
        ransac.fit(outliers[:, 0].reshape(-1, 1), outliers[:, 1])
        inlier_mask = ransac.inlier_mask_

        # Extract inliers and outliers
        inliers = outliers[inlier_mask]
        outliers = outliers[~inlier_mask]

        # Sort inliers by x-coordinate
        inliers = inliers[np.argsort(inliers[:, 0])]

        # Create lines from inliers
        if len(inliers) > 0:
            x1, y1 = inliers[0][0], inliers[0][1]
            x2, y2 = inliers[-1][0], inliers[-1][1]
            lines.append([x1, y1, x2, y2])
            cv.line(img, (x1, 0), (x2, img.shape[0]), (0, 0, 255), 2)

    return lines, img

def linear_regression(img, centers=[], lines=[]):
    centers = np.array(sorted(centers, key=lambda x: x[0]))
    centers = np.array(centers)
    x, y = centers[:, 0], centers[:, 1]
    xy = x*y
    xsq = x**2
    
    n = len(x)
    m = (n*np.sum(xy)-np.sum(x)*np.sum(y))/(n*np.sum(xsq)-np.sum(xsq))
    b = (np.sum(y)*np.sum(xsq)-np.sum(x)*np.sum(xy))/(n*np.sum(xsq)-np.sum(xsq))

    angles = np.degrees(np.arctan(m))


# Calculate angle between ROI and lines
def calculate_angle(lines):
    left_line = lines[0]
    right_line = lines[-1]

    # Calculate left-most and right-most line gradients
    ml = (left_line[3] - left_line[1])/(left_line[2]-left_line[0])
    mr = (right_line[3] - right_line[1])/(right_line[2]-right_line[0])

    # Calculate angles
    angle_left = round(90 - math.degrees(math.atan2(left_line[3] - left_line[1], left_line[2] - left_line[0])), 3)
    angle_right = round(90 - math.degrees(math.atan2(right_line[3] - right_line[1], right_line[2] - right_line[0])), 3)

    return angle_left, angle_right

# Tractor guidance
def tractor_guidance(img, lines):
    # Extract wheel position assuming camera is on the middle of tractor
    pw = img.shape[1]/2

    # Extract most-left and most-right row position
    pl = lines[0]
    pr = lines[-1]
    
    # Calculate distances from the right and left crop row to the wheel
    dl = abs(pw-pl)
    dr = abs(pw-pr)
    
    # Return the control signal
    return dr-dl

# Function to warp image into a bird-eye view
def warp_img(img, points, width, height, inverse = False):
    pt1 = np.float32(points)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    if inverse:
        matrix = cv.getPerspectiveTransform(pt2, pt1)
    else:
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
        cv.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv.FILLED)
    return img

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
        hist_val = np.sum(image[int(image.shape[0]//region):,:], axis=0)

    max_val = np.max(hist_val)
    min_val = min_percentage*max_val

    index_array = np.where(hist_val >= min_val)
    base_point = int(np.average(index_array))
    print(base_point)
    
    if display:
        img_hist = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for i, intensity in enumerate(hist_val):
            cv.line(img_hist, (i, image.shape[0]), (i, image.shape[0]-intensity//255//region), (255, 0, 255), 1)
            cv.circle(img_hist, (base_point, image.shape[0]), 20, (0, 255, 0), cv.FILLED)
        return base_point, img_hist
    return base_point

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass