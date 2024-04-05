import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("jeff.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# equalized based on grayscale values
gray_eq = cv2.equalizeHist(gray)

channels = cv2.split(img)
eq_channels = []

# equalized based on rgb values
for chann in channels:
    eq_channels.append(cv2.equalizeHist(chann))
# THis may mess with the brightnesses

equalized_eq = cv2.merge(eq_channels)


######## HSV equalization

# Instead we can break the image down into "cylindrical coordinates", or HSV
# Here, there is an
# angle, the hue or color
# Height, the saturation/brightness (how black it is)
# Radius, the value (how strong the color is)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

cv2.imshow("hue", h)
cv2.imshow("saturation", s)

cv2.imshow("value", v)
cv2.waitKey(0)
# equalized based on hsv values
# This maintains the hues, but increases contrast
eq_v = cv2.equalizeHist(v)
equalized_hsv = cv2.merge([h, s, eq_v])
equalized_hsv = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)


#########CLAHE equalization
# Equalized based on a tiled grid, so locally the image is equalized
# BUt over the entire image the image is not equalized
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)


cv2.imshow("gray", gray)
cv2.imshow("gray_eq", gray_eq)
cv2.imshow("org", img)
cv2.imshow("eq", equalized_eq)
cv2.imshow("hsv", equalized_hsv)
cv2.imshow("clahe", clahe_img)
cv2.waitKey(0)
