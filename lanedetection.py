import numpy as np
import cv2 as cv
import matplotlib as plt
import seaborn as sns
import moviepy as mpy
import scipy
import utils

if __name__ == '__main__':
    cap = cv.videoCapture('vidi.mp4')
    while True:
        success, img = cap.read()
        img = cv.resize(img, (640, 480))
        cv.imshow('Pi Camera', img)
        cv.waitKey(1)