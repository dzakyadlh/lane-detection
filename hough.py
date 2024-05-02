import numpy as np
import cv2 as cv

def hough_tf(img, ht = 255, hl = 255, st = 255, sl = 255, vt = 255, vl = 255, show = False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, hl, ht, apertureSize=3)
    if(show == True):
        cv.imshow(gray)
        cv.imshow(edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=30, maxLineGap=30)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return img