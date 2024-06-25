import pandas as pd
import numpy as np

def write_to_excel(left_centers, right_centers, lines, slopes):
    # Post-process centers to calculate min and max values for each frame_count
    processed_left_centers = []
    processed_right_centers = []

    # Helper function to extract min and max centers for each frame
    def extract_min_max_for_frame(centers):
        centers_array = np.array(centers)
        max_center_x = np.max(centers_array[:, 1])
        max_center_y = np.max(centers_array[:, 2])
        min_center_x = np.min(centers_array[:, 1])
        min_center_y = np.min(centers_array[:, 2])
        return max_center_x, max_center_y, min_center_x, min_center_y

    # Group centers by frame count and calculate min and max values
    frame_counts_left = set([center[0] for center in left_centers])
    frame_counts_right = set([center[0] for center in right_centers])

    for frame in frame_counts_left:
        frame_centers = [center for center in left_centers if center[0] == frame]
        max_x, max_y, min_x, min_y = extract_min_max_for_frame(frame_centers)
        processed_left_centers.append([frame, max_x, max_y, min_x, min_y])

    for frame in frame_counts_right:
        frame_centers = [center for center in right_centers if center[0] == frame]
        max_x, max_y, min_x, min_y = extract_min_max_for_frame(frame_centers)
        processed_right_centers.append([frame, max_x, max_y, min_x, min_y])

    angles = []

    for slope in slopes:
        left_angle = np.rad2deg(np.arctan2(slope[1],1))
        if left_angle < 0:
            left_angle += 180
        right_angle = np.rad2deg(np.arctan2(slope[2],1))
        if right_angle < 0:
            right_angle += 180
        angles.append([slope[0], left_angle, right_angle])

    # Convert lists to DataFrames
    df_left_centers = pd.DataFrame(processed_left_centers, columns=['frame_count', 'max_center_x', 'max_center_y', 'min_center_x', 'min_center_y'])
    df_right_centers = pd.DataFrame(processed_right_centers, columns=['frame_count', 'max_center_x', 'max_center_y', 'min_center_x', 'min_center_y'])
    df_lines = pd.DataFrame(lines, columns=['frame_count', 'xl1', 'xl2', 'xr1', 'xr2'])
    df_slopes = pd.DataFrame(slopes, columns=['frame_count', 'slope_left', 'intercept_left', 'slope_right', 'intercept_right'])
    df_angles = pd.DataFrame(angles, columns=['frame_count', 'angle_left', 'angle_right'])

    # Write DataFrames to Excel
    with pd.ExcelWriter('output3.xlsx') as writer:
        df_left_centers.to_excel(writer, sheet_name='Left_Centers', index=False)
        df_right_centers.to_excel(writer, sheet_name='Right_Centers', index=False)
        df_lines.to_excel(writer, sheet_name='Lines', index=False)
        df_slopes.to_excel(writer, sheet_name='Slopes', index=False)
        df_angles.to_excel(writer, sheet_name='Angles', index=False)
