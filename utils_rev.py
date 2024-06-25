import numpy as np
import cv2 as cv

# Function for thresholding the image
def thresholding(img, h_max, h_min, v_max, v_min, s_max, s_min):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    return mask

# Draw center points for each bounding boxes
def draw_centers(img, bboxes, roi1=0, roi2=0, color=(255, 0, 255)):
    centers = []
    for bbox in bboxes:
        x, y, w, h, = bbox
        center_x = round(x + w / 2)
        center_y = round(y + h / 2)
        if roi1!=0:
            if center_x > roi2 or center_x < roi1:
                continue
            elif center_y > roi2 or center_y < roi1:
                continue
        cv.circle(img, [center_x, center_y], 5, color, -1)
        centers.append([center_x, center_y])
    return centers, img

def obtain_centers(img, bboxes, roi1=0, roi2=0):
    centers = []
    for bbox in bboxes:
        x, y, w, h = bbox
        center_x = round(x + w / 2)
        center_y = round(y + h / 2)
        
        # Apply region of interest (ROI) filtering
        if roi1 != 0:
            if center_x > roi2 or center_x < roi1 or center_y > roi2 or center_y < roi1:
                continue
        
        centers.append((center_x, center_y))
    
    # Sort centers by y-coordinate
    centers.sort(key=lambda c: c[1])  # Sort by the second element (y-coordinate)

    # Initialize lists for left and right centers
    left_centers = []
    right_centers = []
    
    # Iterate through sorted centers and assign them alternately to left and right
    for i in range(0, len(centers), 2):
        if i < len(centers) - 1:
            if centers[i][0] < centers[i+1][0]:
                left_centers.append(centers[i])
                right_centers.append(centers[i+1])
            else:
                left_centers.append(centers[i+1])
                right_centers.append(centers[i])
        else:
            # If there's an odd number of centers, the last one goes to left_centers
            left_centers.append(centers[i])
    
    return left_centers, right_centers, img

def process_centers(img_center, centers, side, intersect, min_line_length, max_line_gap, min_angle, max_angle):
    left_fit = []
    right_fit = []
    # Handling for lines with infinite slope
    left_inf = []
    right_inf = []

    # Draw centers
    for center in centers:
        cv.circle(img_center, (center[0], center[1]), 5, (255, 0, 255), -1)
    
    # Thresholding
    img_thresh = thresholding(img_center, 150, 150, 255, 255, 255, 255)
    # Canny edge detection
    edges = cv.Canny(img_thresh, 150, 255, apertureSize=3)
    # Hough transform line prediction
    lines = cv.HoughLinesP(edges, 1, np.pi/180, intersect, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        print('No lines detected')
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Checking if the line has infinite slope
        if x1 != x2:
            # Determine the slope and intercept
            m, b = np.polyfit((x1, x2), (y1, y2), 1)

            # Checking if line is vertical
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if angle < 0:
                angle += 180
            if not (min_angle <= angle <= max_angle) and 0 < x1 < img_center.shape[1]:
                continue
            if side == 'left':
                left_fit.append((m,b))
            else:
                right_fit.append((m,b))
        else:
            if side == 'left':
                left_inf.append((x1, x2))
            else:
                right_inf.append((x1, x2))
    if side == 'left':
        return left_fit, left_inf
    else:
        return right_fit, right_inf

# Determine coordinates from slope and intercept
def make_coordinates(img, line_params):
    m, b = line_params
    y1 = img.shape[0]
    y2 = int(y1 * 1/3)
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    return np.array([x1, y1, x2, y2])

# Averaging the slopes and intercept to obtain the optimal line
def process_output(img, fit, inf):
    if fit and len(fit) > len(inf):
        fit_avg = np.average(fit, axis=0)
        line = make_coordinates(img, fit_avg)
    elif inf:
        inf = np.average(inf, axis=0)
        line = np.array([inf[0], img.shape[0], inf[1], int(img.shape[0] * 1/3)])
        fit_avg = [float('inf'), line[0]]
    else:
        line = np.array([0, 0, 0, 0])
        fit_avg = [0, 0]
    return line, fit_avg

# Hough transformation
def hough_transform(img, left_centers, right_centers, intersect, min_line_length, max_line_gap, min_angle, max_angle, show=False):
    img_center_left = np.zeros_like(img)
    left_fit, left_inf = process_centers(img_center_left, left_centers, 'left', intersect, min_line_length, max_line_gap, min_angle, max_angle)
    img_center_right = np.zeros_like(img)
    right_fit, right_inf = process_centers(img_center_right, right_centers, 'right', intersect, min_line_length, max_line_gap, min_angle, max_angle)

    # Averaging the line obtained to get the optimal line
    left_line, left_fit_avg = process_output(img, left_fit, left_inf)
    right_line, right_fit_avg = process_output(img, right_fit, right_inf)

    averaged = np.array([left_line, right_line])
    slopes = np.array([left_fit_avg, right_fit_avg])

    # Draw the line if needed
    line_image = np.zeros_like(img)
    if show:
        for line in averaged:
            if 0 <= line[0] <= img.shape[1]:
                x1, y1, x2, y2 = line.reshape(4)
                cv.line(line_image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 5)

    result_img = cv.addWeighted(img, 0.9, img_center_left, 1, 1)
    result_img = cv.addWeighted(result_img, 0.9, img_center_right, 1, 1)
    result_img = cv.addWeighted(result_img, 0.9, line_image, 1, 1)
    return slopes, averaged, result_img

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