import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

def hough_line(img, angle_st = 1, is_white = True, value_th=150):
    # rho and theta
    theta = np.deg2rad(np.arange(-90.0, 90.0, angle_st))
    w, h = img.shape
    max_dist = int(round(math.sqrt(w*w+h*h)))
    rho = np.linspace(-max_dist, max_dist, max_dist*2)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n_theta = len(theta)

    accumulator = np.zeros((2*max_dist, n_theta), dtype=np.uint8)
    
    is_edge = img > value_th if is_white else img < value_th
    y_idx, x_idx = np.nonzero(is_edge)

    # Voting
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        for t_idx in range(n_theta):
            rho = max_dist + int(round(x*cos_t[t_idx]+y*sin_t[t_idx]))
            accumulator[rho, t_idx] +=1
    
    return accumulator, theta, rho

def show_hough_line(img, accumulator, theta, rho, path = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img)
    ax[0].set_title('Input Image')
    ax[0].axis('image')

    ax[1].imshow(accumulator, cmap = 'jet', extent = [np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]])
    ax[1].set_aspect('equal', adjustable = 'box')
    ax[1].set_title('Hough Transform')
    ax[1].set_xlabel('Angles (deg)')
    ax[1].set_ylabel('Distance (px)')
    ax[1].axis('image')

    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()