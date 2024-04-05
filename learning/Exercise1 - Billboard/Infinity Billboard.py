import cv2
import numpy as np

base_img = cv2.imread("learning/Exercise1 - Billboard/billboard.jpg")
# base_img = cv2.imread("obama_meme.jpg")
# base_img = cv2.resize(base_img, None, fx=0.2, fy=0.2)

# gets sizes of images
(
    base_h,
    base_w,
) = base_img.shape[:2]

# corners of image
src_points = np.array([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]], np.float32)


# corners of billboard
# def onClick(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(dest_points) < 4:
#             dest_points.append([x, y])
#             cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
#             cv2.imshow("Img", img_copy)


# dest_points = []

# cv2.namedWindow("Img", cv2.WINDOW_FREERATIO)

# img_copy = base_img.copy()
# # Gets corners of billboard
# cv2.setMouseCallback("Img", onClick)
# cv2.imshow("Img", img_copy)
# cv2.waitKey(0)


# dest_float = np.array(dest_points, dtype=np.float32)

# dest_float = np.array(dest_points, dtype=np.float32)
# dest_float = np.array([[64, 90], [62, 477], [531, 410], [534, 149]], dtype=np.float32)
dest_float = np.array(
    [[315.0, 450.0], [310.0, 2381.0], [2665.0, 2061.0], [2675.0, 745.0]],
    dtype=np.float32,
)

# creates transformation array
M = cv2.getPerspectiveTransform(src_points, dest_float)

for i in range(20):

    img_copy = base_img.copy()

    cv2.namedWindow("Img", cv2.WINDOW_FREERATIO)
    warped_img = cv2.warpPerspective(img_copy, M, (base_w, base_h))

    # Fills in the billboard with black <- Charlie's  solution
    base_img = cv2.fillPoly(
        base_img, [dest_float.astype(int)], (0, 0, 0), lineType=cv2.LINE_AA
    )
    base_img = cv2.bitwise_or(warped_img, base_img)


cv2.imshow("Img", base_img)
cv2.waitKey(0)
