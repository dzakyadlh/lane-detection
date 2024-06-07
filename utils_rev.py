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
    corners = [[[x, y], [x + w, y]] for x, y, w, h in bboxes]
    
    # Sort corners based on the x-coordinate of the top-left corner
    corners.sort(key=lambda corner: corner[0][0])
    
    # Step 2: Iterate through and check the distance of each x. If the distance < threshold, then its on the same line, else not.
    left_line = []
    right_line = []
    
    pivot = corners[0][0][0]
    for corner in corners:
        top_left, top_right = corner
        if (abs(pivot - top_right[0]) < threshold):
            left_line.append(top_right)
        else:
            right_line.append(top_left)
    
    # Step 3: Draw all the remaining coordinates as points using cv.circle
    draw_points(img, left_line)
    draw_points(img, right_line)

    # cv.imshow('img', img)
    # cv.waitKey(0)
    
    # Step 4: Return left_line, right_line, and img
    return left_line, right_line, img

# Function for thresholding the image
def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask


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

def make_coordinates(img, line_params):
    m, b = line_params
    y1 = img.shape[0]
    y2 = int(y1*(1/3))
    x1 = int((y1 - b)/m)
    x2 = int((y2 - b)/m)
    return np.array([x1, y1, x2, y2])

def probabilistic_hough_transform2(img, intersect, min_line_length, max_line_gap, min_angle, max_angle, max_xgap):
    img_thresh = thresholding(img, 150, 150, 255, 255, 255, 255)
    # cv.imshow('img',img_thresh)
    # cv.waitKey(0)
    edges = cv.Canny(img_thresh, 150, 255, apertureSize=3)
    # cv.imshow('img',edges)
    # cv.waitKey(0)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, intersect, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    left_fit = []
    right_fit = []
    center_x = img.shape[1] // 2
    for line in lines:
        x1, y1, x2, y2 =  line.reshape(4)
        angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        if angle < 0:
            angle+=180
        if angle >= min_angle and angle <= max_angle:
            params = np.polyfit((x1, x2), (y1, y2), 1)
            m = params[0]
            b = params[1]
            midpoint_x = (x1 + x2) / 2
            if midpoint_x < center_x:
                left_fit.append((m, b))
            else:
                right_fit.append((m, b))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_avg)
    right_line = make_coordinates(img, right_fit_avg)
    averaged = np.array([left_line, right_line])
    slopes = np.array([left_fit_avg, right_fit_avg])

    line_image = np.zeros_like(img)
    if averaged is not None:
        for line in averaged:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    result_img = cv.addWeighted(img, 0.8, line_image, 1, 1)
    return slopes, averaged, result_img


# Tractor guidance
def tractor_guidance2(img, slopes):
    print(slopes)
    # Extract the slopes
    ml = round(slopes[0, 0], 3)
    mr = round(slopes[1, 0], 3)

    # Calculate the difference of slopes
    dm = round(abs(ml)-abs(mr), 3)

    cv.putText(img, str(ml), (10, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    cv.putText(img, str(dm), (150, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    cv.putText(img, str(mr), (300, 400), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    
    # Return the control
    return ml, mr, dm, img

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
        angles = np.rad2deg(np.arctan2(lines[:, 3]-lines[:, 1], lines[:, 2]-lines[:, 0]))
        for angle in angles:
            if angle < 0:
                angle+=180
        print(angles)
        mask = (angles >= min_angle) & (angles <= max_angle)
        lines = lines[mask]

        for line in lines:
            cv.line(img, (line[0], line[1]), (line[2], line[3]), color, 2)

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