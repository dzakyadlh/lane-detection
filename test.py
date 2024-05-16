import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = np.zeros((416,416))
cv.line(img,(x1, y1),(x2,y2),(255, 0, 0), 2)