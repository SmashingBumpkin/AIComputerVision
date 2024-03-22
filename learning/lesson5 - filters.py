import cv2
import numpy as np


img = cv2.imread("night-sky3.jpg")
scale = 0.3
img = cv2.resize(img, None, fx=scale, fy=scale)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


######################
# kernel filtering
######################
"""
my_kernel = np.array([[6, 3, 0], [-3, 0, 3], [0, 3, 0]])
filtered_img = cv2.filter2D(img, -1, my_kernel)
# ddepth is number of channels on output
# Setting it to -1 keeps the number of channels same as on input
"""


######################
# BLURRING
######################
"""
filtered_img = cv2.blur(img, (1, 109))  # blur in y and x directions

filtered_img = cv2.GaussianBlur(img, (5, 5), 0)  # ????

filtered_img = cv2.medianBlur(img, 3)
# blur square using the median value

# blurs while remaining sharp
filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
#
"""


######################
# SHARPENING
######################
# No built in function because it's not a normal thing to do
"""
# Sharpen mask - blurs image then compares to original to figure out edges
blurred_img = cv2.GaussianBlur(img, (5, 5), 10)  # 3rd value is sdev
filtered_img = cv2.addWeighted(img, 1, blurred_img, 0.2, 0)
# adds to images with a weighting. gamma changes the color

#Famous kernel transform that works pretty ok at sharpening
my_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
filtered_img = cv2.filter2D(img, -1, my_kernel)
"""


######################
# COUNTOUR EXTRACTION
######################
# Using the sobel derivative, which creates a contour based on changes in colour
"""
sobel_der_x = cv2.Sobel(img_grey, -1, 1, 0)
sobel_der_y = cv2.Sobel(img_grey, -1, 0, 1)
scaled_x = cv2.convertScaleAbs(sobel_der_x)
scaled_y = cv2.convertScaleAbs(sobel_der_y)
filtered_img = cv2.addWeighted(scaled_x, 0.5, scaled_y, 0.5, 0)

der = cv2.Laplacian(img_grey, -1, (10, 10))
filtered_img = cv2.convertScaleAbs(der)
"""

######################
# CARTOONIZATION
######################
# Light blur to clean the image
img_grey = cv2.medianBlur(img_grey, 7)
# extract contours to 8 bit with 5 wide kernel
edges = cv2.Laplacian(img_grey, cv2.CV_8U, ksize=5)
# clean edge image (keep only the strong edges) by applying a threshold
# THRESH_BINARY_INV means background black and contours white
ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
color_img = cv2.bilateralFilter(img, 10, 250, 250)  # d= dimension of kernel
# COnvert grayscale image to colour (ie give it 3 channels)
skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
# join images to make cartoon??
# filtered_img = cv2.addWeighted(color_img, 0.8, skt, 0.2, 0)
filtered_img = cv2.bitwise_and(color_img, skt)
# filtered_img = skt

cv2.imshow("ORIGINAL", img)
cv2.imshow("Image", filtered_img)
cv2.imshow("contour", skt)
cv2.waitKey(0)
