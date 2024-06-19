import math
import numpy as np
import cv2 as cv

# Function for thresholding the image
def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask = cv.inRange(hsv, lower, upper)
    return mask

# Draw center points for each bounding box
def draw_centers(img, bboxes, roi1=0, roi2=0, color=(255, 0, 255)):
    centers = []
    h, w = img.shape[:2]
    for x, y, w, h in bboxes:
        center_x = round(x + w / 2)
        center_y = round(y + h / 2)
        if roi1 != 0 and (center_x > roi2 or center_x < roi1 or center_y > roi2 or center_y < roi1):
            continue
        cv.circle(img, (center_x, center_y), 5, color, -1)
        centers.append([center_x, center_y])
    return centers, img

def make_coordinates(img, line_params):
    m, b = line_params
    y1 = img.shape[0]
    y2 = int(y1 / 3)
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    return np.array([x1, y1, x2, y2])

def hough_transform(img, intersect, min_line_length, max_line_gap, min_angle, max_angle, show=False):
    img_thresh = thresholding(img, 150, 150, 255, 255, 255, 255)
    edges = cv.Canny(img_thresh, 150, 255, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, intersect, minLineLength=min_line_length, maxLineGap=max_line_gap)

    left_fit = []
    right_fit = []
    center_x = img.shape[1] // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 != x2:
                m, b = np.polyfit((x1, x2), (y1, y2), 1)
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                angle = angle + 180 if angle < 0 else angle
                if not (min_angle <= angle <= max_angle):
                    continue

                midpoint_x = (x1 + x2) / 2
                (left_fit if midpoint_x < center_x else right_fit).append((m, b))

    left_line = np.mean([make_coordinates(img, line) for line in left_fit], axis=0) if left_fit else None
    right_line = np.mean([make_coordinates(img, line) for line in right_fit], axis=0) if right_fit else None

    line_image = np.zeros_like(img)
    if show and left_line is not None and right_line is not None:
        for line in [left_line, right_line]:
            x1, y1, x2, y2 = map(int, line)
            cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    result_img = cv.addWeighted(img, 0.8, line_image, 1, 1)
    return (np.array([left_fit, right_fit]), np.array([left_line, right_line])), result_img

# Tractor guidance
def tractor_guidance(img, lines, threshold, show=False):
    xl1, yl1, xl2, yl2 = lines[0]
    xr1, yr1, xr2, yr2 = lines[1]

    center = img.shape[1] // 2
    dl = abs(center - xl2)
    dr = abs(xr2 - center)
    dm = dr - dl

    guide = 0
    color = (0, 0, 0)
    if dm > threshold: 
        guide = -1
        color = (66, 197, 245)
    elif dm < -threshold: 
        guide = 1
        color = (66, 197, 245)
    else: 
        guide = 0
        color = (245, 197, 66)
    
    if show:
        cv.line(img, (center, img.shape[0]), (center, img.shape[0] - 40), (0, 0, 0), 3)
        cv.putText(img, str(dl), (10, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv.putText(img, str(dm), (150, 400), cv.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv.putText(img, str(dr), (300, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv.imshow('img', img)
        cv.waitKey(0)
    else:
        print(guide)
    
    return dl, dr, dm, guide, img


# Tractor guidance
def tractor_guidance2(img, slopes, threshold):
    # Extract the slopes
    ml = round(slopes[0, 0], 3)
    mr = round(slopes[1, 0], 3)

    # Calculate the difference of slopes
    dm = round(abs(ml)-abs(mr), 3)

    cv.putText(img, str(ml), (10, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    if dm > threshold:
        cv.putText(img, str(dm), (150, 400), cv.FONT_HERSHEY_COMPLEX, 1, (245, 197, 66), 2)
    elif dm < -threshold:
        cv.putText(img, str(dm), (150, 400), cv.FONT_HERSHEY_COMPLEX, 1, (168, 164, 50), 2)
    else:
        cv.putText(img, str(dm), (150, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    cv.putText(img, str(mr), (300, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    
    # Return the control
    return ml, mr, dm, img

# Function to draw circles at the specified points
def draw_points(img, points):
    # Round each coordinate in points
    points = [[round(coord) for coord in point] for point in points]
    for point in points:
        cv.circle(img, tuple(point), 5, (255, 0, 255), -1)

# Function to process bounding boxes
def process_bboxes(img, bboxes, threshold=30):
    # Keep the top-left and top-right coordinates as an array
    corners = [[[x, y], [x + w, y]] for x, y, w, h in bboxes]
    
    # Sort corners based on the x-coordinate of the top-left corner
    corners.sort(key=lambda corner: corner[0][0])
    
    # Iterate through and check the distance of each x. If the distance < threshold, then its on the same line, else not.
    left_line = []
    right_line = []
    
    pivot = corners[0][0][0]
    for corner in corners:
        top_left, top_right = corner
        if (abs(pivot - top_right[0]) < threshold):
            left_line.append(top_left)
        else:
            right_line.append(top_right)
    
    # Draw all the remaining coordinates as points using cv.circle
    draw_points(img, left_line)
    draw_points(img, right_line)
    
    # Return left_line, right_line, and img
    return left_line, right_line, img

def lines_linearization(img, left_line, right_line):
    left_line = np.array(left_line)
    right_line = np.array(right_line)

    # Sorting lines based on y
    left_line = left_line[np.argsort(left_line[:, 1])]
    right_line = right_line[np.argsort(right_line[:, 1])]

    # Calculate slopes of each lines
    ml = (left_line[-1, 1] - left_line[0, 1]) / (left_line[-1, 0] - left_line[0, 0] if left_line[-1, 0] - left_line[0, 0] != 0 else 1)
    mr = (right_line[-1, 1] - right_line[0, 1]) / (right_line[-1, 0] - right_line[0, 0] if right_line[-1, 0] - right_line[0, 0] != 0 else 1)

    cv.line(img, (left_line[0, 0], 0), (round(left_line[0, 0]+img.shape[0]/ml), img.shape[0]), (0, 0, 255), 2)
    cv.line(img, (right_line[0, 0], 0), (round(right_line[-1, 0]+img.shape[0]/mr), img.shape[0]), (0, 0, 255), 2)

    return [left_line[0, 0], left_line[-1, 0]], [right_line[0, 0], right_line[-1, 0]], img