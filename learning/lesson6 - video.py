import cv2
import numpy as np

cap = cv2.VideoCapture(0) # 0 streams from the first webcam connected
while True:
    ret, img = cap.read() # returns bool if succesfully got frame, and the frame

    scale = 1
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    cv2.imshow("Image", filtered_img)
    k = cv2.waitKey(10)
    
    if k == ord('q'):
        break
