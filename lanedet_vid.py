import cv2 as cv
import time
import utils_rev
import yolov4
import data_to_excel

# Function to process a single frame
def process_frame(frame, model):
    # Resize frame
    frame_resized = cv.resize(frame, (640, 640))

    # Run detection with yolov4
    labels, scores, bboxes = yolov4.predict(frame_resized, model, 0.5)

    # Obtain centers
    left_centers, right_centers, frame_centers = utils_rev.obtain_centers(frame_resized, bboxes)
    print(left_centers)

    # Run hough transform
    slopes, averaged_line, frame_hough = utils_rev.hough_transform(frame_centers, left_centers, right_centers, 10, 10, 80, 60, 120, show=True)

    return left_centers, right_centers, averaged_line, slopes, frame_hough

# Take input
cap = cv.VideoCapture('assets/videos/finaltest30.mp4')

# Read network model
model_file = 'yolo_archive/models/yolov4/v4/yolov4-obj_best.weights'
config_file = 'yolo_archive/yolov4-obj.cfg'
net = cv.dnn.readNetFromDarknet(config_file, model_file)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
model = cv.dnn.DetectionModel(net)
model.setInputParams(size=(640, 640), scale=1/255, swapRB=True, crop=False)

frame_count = 1
wait_key = 1
start_time = time.time()

# Initialize lists to store centers and slopes
all_left_centers = []
all_right_centers = []
all_lines = []
all_slopes = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Process frame
        left_centers, right_centers, lines, slopes, frame_final = process_frame(frame, model)

        # Store centers and slopes with frame count
        for center in left_centers:
            all_left_centers.append([frame_count, center[0], center[1]])
        for center in right_centers:
            all_right_centers.append([frame_count, center[0], center[1]])
        if lines is not None:
            all_lines.append([frame_count, lines[0][0], lines[0][2], lines[1][0], lines[1][2]])
        else:
            all_lines.append([frame_count, '-', '-', '-', '-'])
        if slopes is not None:
            all_slopes.append([frame_count, slopes[0][0], slopes[0][1], slopes[1][0], slopes[1][1]])
        else:
            all_slopes.append([frame_count, '-', '-'])
            all_slopes.append([frame_count, '-', '-'])

        # Display the resulting frame
        cv.imshow('frame', frame_final)


        # Increment frame counter
        frame_count += 1

        if cv.waitKey(wait_key) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
cv.destroyAllWindows()

# Calculate FPS
elapsed_time = time.time() - start_time
fps = frame_count / elapsed_time
print('FPS: ' + str(fps))

# Write processed data to Excel
# data_to_excel.write_to_excel(all_left_centers, all_right_centers, all_lines, all_slopes)