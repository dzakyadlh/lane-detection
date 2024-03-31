import numpy as np
import cv2 as cv
import utils

def lane_det(frame, display = 2):
    result = frame.copy()

    threshold_frame = utils.thresholding(frame, 0, 0, 62, 179, 75, 110)

    h, w, c = frame.shape
    points = utils.get_trackbar_points()
    warped_frame = utils.warp_img(threshold_frame, points, w, h)
    warp_points = utils.draw_points(frame, points)

    mid_point, hist_img = utils.get_histogram(warped_frame, display=True, min_percentage=0.5, region=4)
    base_point, hist_img = utils.get_histogram(warped_frame, display=True, min_percentage=0.9)
    curve = base_point-mid_point

    curves = []
    curves.append(curve)
    if len(curves)>10:
        curves.pop(0)
    curve = int(sum(curves)/len(curves))

    if display!=0:
        inv_warped = utils.warp_img(warped_frame, points, w, h, inverse=True)
        inv_warped = cv.cvtColor(inv_warped, cv.COLOR_GRAY2BGR)
        inv_warped[0:h//3, 0:w] = 0, 0, 0
        lane_color = np.zeros_like(frame)
        lane_color[:] = 0, 255, 0
        lane_color = cv.bitwise_and(inv_warped, lane_color)
        result = cv.addWeighted(result, 1, lane_color, 1, 0)
        mid_y = 450
        cv.putText(result, str(curve), (w//2-80, 85), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
        cv.line(result, (w // 2,mid_y), (w // 2 + (curve * 3),mid_y), (255, 0, 255), 5)
        cv.line(result, ((w // 2 + (curve * 3)),mid_y - 25), (w // 2 + (curve * 3),mid_y + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = w // 20
            cv.line(result, (w * x + int(curve // 50),mid_y - 10),
                     (w * x + int(curve // 50),mid_y + 10), (0, 0, 255), 2)
        #fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
        #cv.putText(result, 'FPS ' + str(int(fps)), (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        frame_stacked = utils.stackImages(0.7, ([frame, warp_points, warped_frame],
                                             [hist_img, lane_color, result]))
        cv.imshow('Frame Stacked', frame_stacked)
    elif display == 1:
        cv.imshow('Result', result)


    #### NORMALIZATION
    curve = curve/100
    if curve>1: curve ==1
    if curve<-1:curve == -1
    
    return curve

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
        curve = lane_det(frame)
        print(curve)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()