import math
import numpy as np
import cv2 as cv

# Draw center points for each bounding boxes
def draw_centers(img, bboxes, color):
    for bbox in bboxes:
        x, y, w, h, = bbox
        center_x = round(x + w / 2)
        center_y = round(y + h / 2)
        cv.circle(img, [center_x, center_y], 5, color, -1)
    return img

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
    img_thresh = thresholding(img, 150, 150, 255, 255, 255, 255)
    edges = cv.Canny(img_thresh, 150, 255, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, intersect, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:

        # Extract each coordinates
        for line in lines:
            for coordinates in line:
                x1, y1, x2, y2 = coordinates
                # Making sure the line go through the y1 = 0 to y2 = max y
                y1 = 0
                y2 = img.shape[0]

                # Making sure the lines are vertical
                angle = np.arctan2(y2-y1, x2-x1)*180.0/np.pi
                if angle >= min_angle and angle <= max_angle:
                    hough_lines.append([x1, y1, x2, y2])

    # Averaging the lines
    hough_avg = []
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
            if i == n_hough-2:
                avg //= n
                hough_avg.append(avg)
                cv.line(img,(avg[0],avg[1]),(avg[2],avg[3]),color,2)

    return hough_avg, img

# Calculate angle between ROI and Hough lines
def calculate_angle(hough_lines):
    # Calculate left-most and right-most Hough lines gradients
    n = len(hough_lines)
    line = hough_lines[n-1]
    m = (line[3]-line[1])/(line[2]-line[0])

    # Direction of the turn, 0 for left, 1 for right
    turn_dir = 0
    if m > 0:
        turn_dir = 1

    # Calculate angle
    angle = round(abs(math.atan(m)), 3)

    return angle, turn_dir

# Function for thresholding the image
def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask

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