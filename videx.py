import cv2
cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))

cap.set(3, 1280)
cap.set(4, 720)

# recorded = cv2.VideoWriter('recorded_vid.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # recorded.write(frame)
        
        frame = cv2.rectangle(frame, (200,200), (500,500), (255, 0, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
# recorded.release()
cv2.destroyAllWindows()