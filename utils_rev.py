import math
import numpy as np
import cv2 as cv

# Function to draw circles at the specified points
def draw_points(img, points):
    for point in points:
        cv.circle(img, tuple(point), 5, (255, 0, 255), -1)

# Function to process bounding boxes and extract lines
def process_bboxes(img, bboxes, threshold=30):
    # Step 1: Keep the top-left and top-right coordinates as an array
    centers = [[[x, y], [x + w, y]] for x, y, w, h in bboxes]
    
    # Step 2: Iterate through and check the distance of each x. If the distance < threshold, then its on the same line, else not.
    left_line = []
    right_line = []
    
    pivot = centers[0][0][0]
    for center in centers:
        top_left, top_right = center
        if (abs(pivot - top_right[0]) < threshold):
            left_line.append(top_left)
        else:
            right_line.append(top_right)
    
    # Step 3: Draw all the remaining coordinates as points using cv.circle
    draw_points(img, left_line)
    draw_points(img, right_line)
    
    # Step 4: Return left_line, right_line, and img
    return left_line, right_line, img

# Function for thresholding the image
def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask

def probabilistic_hough_transform(img, intersect, min_line_length, max_line_gap, min_angle, max_angle, max_xgap, color=(0, 0, 255)):
    img_thresh = thresholding(img, 150, 150, 255, 255, 255, 255)
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

        # for line in lines:
        #     cv.line(img, (line[0], line[1]), (line[2], line[3]), color, 2)

        if len(lines) > 0:
            # Sorting lines based on x1
            lines = lines[np.argsort(lines[:, 0])]

            # Calculating the slope of each lines
            left_slopes = np.array([])
            right_slopes = np.array([])
            pivot = lines[0, 0]
            x1_left = lines[0, 0]
            i=0
            for line in lines:
                x1, y1, x2, y2 = line
                if abs(pivot-line[0]) < max_xgap:
                    if (x2-x1) != 0:
                        left_slopes = np.append(left_slopes, (y2-y1)/(x2-x1))
                    i+=1
                else:
                    break
            x1_right = lines[i, 0] if i < len(lines) else x1_left
            for j in range(i, len(lines)):
                x1, y1, x2, y2 = lines[j]
                if (x2-x1) != 0:
                    right_slopes = np.append(right_slopes, (y2 - y1)/(x2 - x1))

            # Calculating the average slope for each lines
            avg_left_slope = np.mean(left_slopes)
            avg_right_slope = np.mean(right_slopes)

            print('m_left = ' + str(avg_left_slope))
            print('m_right = ' + str(avg_right_slope))

            # Drawing the lines
            cv.line(img, (x1_left, 0), (round(x1_left+img.shape[0]/avg_left_slope), img.shape[0]), (255, 0, 0), 1)
            cv.line(img, (x1_right, 0), (round(x1_right+img.shape[0]/avg_right_slope), img.shape[0]), (255, 0, 0), 1)

    return lines, img


# Tractor guidance
def tractor_guidance(img, left_line, right_line):
    # Extract camera's mid position assuming camera is on the middle of tractor
    xc = img.shape[1]/2
    # Assuming yc
    yc = 350

    xl1, xl2 = left_line
    xr1, xr2 = right_line

    # Calculate slopes of each lines
    ml = img.shape[0]/(xl2-xl1)
    mr = img.shape[0]/(xr2-xr1)
    
    # Calculate the difference of slopes
    dm = round(abs(ml)-abs(mr), 3)
    
    # Return the control signal
    return dm


# Linearization method
def lines_linearization(img, left_line, right_line):

    left_line = np.array(left_line)
    right_line = np.array(right_line)

    # Sorting lines based on y
    left_line = left_line[np.argsort(left_line[:, 1])]
    right_line = right_line[np.argsort(right_line[:, 1])]

    cv.line(img, (left_line[0, 0], 0), (left_line[-1, 0], img.shape[0]), (0, 0, 255), 2)
    cv.line(img, (right_line[0, 0], 0), (right_line[-1, 0], img.shape[0]), (0, 0, 255), 2)

    return [left_line[0, 0], left_line[-1, 0]], [right_line[0, 0], right_line[-1, 0]], img


def lines_linearization2(img, left_line, right_line):
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