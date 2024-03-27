import numpy as np
import cv2
import matplotlib.pyplot as plt

####### WEIRD ADDITION 

x = np.uint8([250])
y = np.uint8([50])

result_opencv = cv2.add(x,y)
result_np = x + y

print(f'x {x} y {y}')

print(f'Opencv: {result_opencv}') # wtf
print(f'Nnumpy: {result_np}') # overflows

####### ADDING IMAGES

img = cv2.imread('jeff.png')
M = np.full(img.shape, -10, dtype='uint8')

added_img = cv2.add(img, M)

# cv2.imshow("img",img)
# cv2.imshow("added_img",added_img)
# cv2.waitKey(1000)


######### ADDING + resizing images

img2 = cv2.imread('night-sky2.jpg')
img2_res = cv2.resize(img2, (img.shape[1],img.shape[0]))

added_img = cv2.add(img, img2_res)

# cv2.imshow("added_img",added_img)
# cv2.waitKey(0)

###### bitwise ops

img1 = np.zeros((500,500), dtype = 'uint8')
img2 = np.zeros((500,500), dtype = 'uint8')

cv2.rectangle(img1, (200,100),(400,200), (255,255,255),-1)
cv2.rectangle(img2, (50,50),(300,300), (255,255,255),-1)

result = cv2.bitwise_and(img1,img2)
result1 = cv2.bitwise_or(img1,img2)
result2 = cv2.bitwise_xor(img1,img2)
result3 = cv2.bitwise_not(img1,img2)

# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)
# cv2.imshow("res",result)
# cv2.imshow("res1",result1)
# cv2.imshow("res2",result2)
# cv2.imshow("res3",result3)
# cv2.waitKey(1000)

########## masks and inversions

mask = np.zeros(img.shape,dtype='uint8')
cv2.rectangle(mask,(100,50),(350,350),(255,255,255),-1)
result = cv2.bitwise_xor(img,mask)
# cv2.imshow("masked",result)
# cv2.waitKey(0)

####### histograms


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray],[0],None,[256],[0,256])

plt.plot(hist)
# plt.show()

color = ('b','g','r')
for i,col in enumerate(color):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist, color = col)
# plt.show()

######## equalize hist

gray_eq = cv2.equalizeHist(gray)
cv2.imshow("equalized",gray_eq)
cv2.waitKey(0)
hist = cv2.calcHist([gray_eq],[0],None,[256],[0,256])

plt.plot(hist)
plt.show()