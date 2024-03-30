import os
import numpy
import cv2 as cv
import utils

def lane_det(frame):
    threshold_frame = utils.thresholding(frame, 0, 0, 62, 36, 54, 110)

    h, w, c = frame.shape
    points = utils.get_trackbar_points()
    warped_frame = utils.warp_img(threshold_frame, points, w, h)

    base_point, hist_img = utils.get_histogram(warped_frame, display=True, min_percentage=0.5, region=1/4)

    cv.imshow("result", warped_frame)
    cv.imshow('histogram', hist_img)
    
    return None

if __name__=='__main__':
    cap = cv.VideoCapture("./assets/videos/road_vid.mp4")
    warp_points = [118, 175, 66, 232]
    utils.initialize_points_trackbars(warp_points)
    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        if cap.get(cv.CAP_PROP_FRAME_COUNT) == frame_count:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

        ret, frame = cap.read()
        frame = cv.resize(frame, (480, 240))
        lane_det(frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()