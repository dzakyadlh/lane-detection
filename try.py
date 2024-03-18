import numpy as np
import cv2 as cv
import matplotlib as plt
import seaborn as sns
import moviepy as mpy
import scipy
import utils

arr = np.array([[1,2,3], [4,5,6]])
arr2 = np.array([[10,11,12], [13,14,15]])

filter_arr = arr % 2 == 0
new_arr = arr[filter_arr]

print(np.concatenate((arr, arr2)).reshape(-1))
print(new_arr)
for x in np.nditer(arr[:, ::2]):
    print(x)
