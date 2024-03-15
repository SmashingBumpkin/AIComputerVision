import cv2
import numpy as np

img = cv2.imread(
    "pannone.jpg", cv2.IMREAD_COLOR
)  # second argument optional, allows you to speicfy how file is read
# Note for pngs, the alpha (transparency) channel doesn't have any information, but if you need it, it's -1
print(f"Width: {img.shape[1]} pixels")
print(f"Height: {img.shape[0]} pixels")
print(f"Shape: {img.shape}")

# cv2.imshow('PANNNNAONEONEENENEEEEENEEEEEEEE',img)
# cv2.waitKey(80)
cv2.destroyAllWindows()  # closes image windows in jupyter notebook
cv2.imwrite("jeff.png", img)  # writes new image

(b, g, r) = img[345, 345]  # notice that it's not rgb, it's bgr
print(b, g, r)

# img = np.zeros((500,500,3),dtype='uint8') # creates 500 x 500 x 3 channel array of uints

# GREEN = (0,255,0)
# BLUE = (255,0,0)

# cv2.line(img, (10,10), (200,200), GREEN, thickness=5) # draw a line
# cv2.rectangle(img,(30,30),(70,70), BLUE, thickness=-1) # draw a rectangle, -ve fills it
# cv2.circle(img,(250,250),100,RED,thickness=5)

# cv2.imshow('Img',img)
# cv2.waitKey(5000)
img[270:330, 150:450, 0:2] = 0  # removes blue and green channels
jeff = img[200:400, 120:480]
RED = (0, 0, 255)
cv2.rectangle(jeff, (0, 0), (360, 200), RED, thickness=5)
YELLOW = (0, 255, 255)
cv2.circle(jeff, (100, 99), 5, YELLOW, thickness=-1)
cv2.circle(jeff, (205, 96), 5, YELLOW, thickness=-1)
cv2.line(jeff, (205, 96), (250, 250), YELLOW, thickness=2)  # draw a line
cv2.line(jeff, (100, 99), (250, 250), YELLOW, thickness=2)  # draw a line

# cv2.imshow('PANNNNAONEONEENENEEEEENEEEEEEEE',jeff)
# cv2.waitKey(100)
cv2.imwrite("jeff.png", img)  # writes new image
(bluChan, grnChan, redChan) = cv2.split(img)

img_copy = cv2.merge((grnChan, redChan, bluChan))
# cv2.imshow('copy of PANNNNAONEONEENENEEEEENEEEEEEEE',img_copy)
# cv2.waitKey(1000)

# b = img[:,:,0]
# cv2.imshow('no rd chan',b)
# cv2.waitKey(2000)
print(img.shape[0], img.shape[1], img.shape[2])
upscale_img = cv2.resize(
    img, (img.shape[0] // 3, img.shape[1] * 1)
)  # resizes image, requires ints

upscale_img = cv2.resize(upscale_img, None, fx=1.4, fy=0.7)  # resizes image
tx = 100  # translation
ty = -100  # translation
# stretces and rotates
A11 = 1
A12 = 0.3
A21 = -0.2
A22 = 0.9
# rows and columns
rows = img.shape[0]
cols = img.shape[1]
M = np.float32([[A11, A12, tx], [A21, A22, ty]])
M = cv2.getRotationMatrix2D((cols // 2, rows // 2), 136.2, 0.86)
dst_img = cv2.warpAffine(img, M, (rows, cols))
# cv2.imshow('modified', dst_img)
# cv2.waitKey(2000)

pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])

M = cv2.getAffineTransform(pts_1, pts_2)
# getAffineTransform gets the matrix necessary to translate
# the image where the pixel at posn pts_1 moves to pts_2
print(M)
dst_img2 = cv2.warpAffine(img, M, (rows, cols))
cv2.imshow("modified", dst_img2)
cv2.waitKey(2000)
