import cv2 as cv

cap = cv.VideoCapture(0)

def capture_video(display = False, size = [480, 240]):
    _, frame = cap.read()
    frame = cv.resize(frame, (size[0], size[1]))
    if display:
        cv.imshow('Video',  frame)
    return frame

if __name__ == '__main__':
    while True:
        frame = capture_video(True)