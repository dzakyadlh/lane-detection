import numpy as np
import cv2

img = cv2.imread('test2.jpg', 1)
img = cv2.resize(img, (500, 500))

mask = np.zeros([500,500,3],np.uint8)

print(img.shape)
print(img.size)
print(img.dtype)

# here
# img = cv2.line(img, (0,0), (255,255), (0,0,255), 2)
img  = cv2.rectangle(mask, (100,100), (400,400), (0,0,255), 2)

cv2.imshow('image', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()