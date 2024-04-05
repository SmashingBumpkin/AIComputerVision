import cv2
import numpy as np

base_img = cv2.imread("learning/Exercise1 - Billboard/billboard.jpg")
# base_img = cv2.resize(base_img, None, fx=0.2, fy=0.2)

img_copy = base_img.copy()

img2 = cv2.imread("jeff.png")


# Takes 4 points on an image and puts the points in an array
def onClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dest_points) < 4:
            dest_points.append([x, y])
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Img", img_copy)


dest_points = []

cv2.namedWindow("Img", cv2.WINDOW_FREERATIO)

# Gets corners of billboard
cv2.setMouseCallback("Img", onClick)
cv2.imshow("Img", img_copy)
cv2.waitKey(0)

# gets sizes of images
(
    base_h,
    base_w,
) = base_img.shape[:2]
img2_h, img2_w = img2.shape[:2]

# corners of image
src_points = np.array([[0, 0], [0, img2_h], [img2_w, img2_h], [img2_w, 0]], np.float32)
# corners of billboard
dest_float = np.array(dest_points, dtype=np.float32)
# dest_float = np.array([[64, 90], [62, 477], [531, 410], [534, 149]], dtype=np.float32)

# creates transformation array
M = cv2.getPerspectiveTransform(src_points, dest_float)
warped_img = cv2.warpPerspective(img2, M, (base_w, base_h))

# Fills in the billboard with black <- Charlie's  solution
base_img = cv2.fillPoly(
    base_img, [dest_float.astype(int)], (0, 0, 0), lineType=cv2.LINE_AA
)
combined_img = cv2.bitwise_or(warped_img, base_img)

# Fills in the billboard with black <- Prof's solution
mask = np.zeros(base_img.shape, dtype=np.uint8)
cv2.fillConvexPoly(mask, dest_float.astype(int), (255, 255, 255))
mask = cv2.bitwise_not(mask)
masked_billboard = cv2.bitwise_and(base_img, mask)
final_img = cv2.bitwise_or(masked_billboard, warped_img)

cv2.imshow("Img", combined_img)
cv2.waitKey(0)
cv2.imwrite("PannoneBill.jpg", combined_img)
