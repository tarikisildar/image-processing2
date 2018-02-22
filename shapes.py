import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ek.jpeg')
mask = np.zeros(img.shape[:2],np.uint8)

lower_red = np.array([0,0,200])
upper_red = np.array([100,100,255])

lower_green = np.array([0,120,10])
upper_green = np.array([100,255,100])

maskg = cv2.inRange(img,lower_green,upper_green)
resg = cv2.bitwise_and(img, img,mask=maskg)

mask = cv2.inRange(img,lower_red,upper_red)
res = cv2.bitwise_and(img, img,mask=mask)

gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
ret,thresh = cv2.threshold(gray,150,255,100)
img2,contours,hierarchy =cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

lis=[]
area = 0
for i in contours:
    M = cv2.moments(i)
    lis.append([int(M['m10'] / M['m00']),int(M['m01'] / M['m00'])])
    area+=cv2.contourArea(i)
print("area of red regions:",area)

gray = cv2.cvtColor(resg,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
ret,thresh = cv2.threshold(blur,150,255,100)
img2,contours,hierarchy =cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(0,0,255))
area = 0

li=[]
for i in contours:
    M=cv2.moments(i)
    li.append([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
    area+=cv2.contourArea(i)
print("Area of green regions:",area)
lis=li+lis
lis=sorted(lis,key=lambda x: x[1],reverse=True)
for e in range(len(lis)):
    print("shape",str(e+1)+":",str(lis[e][0])+","+str(lis[e][1]))

#cv2.imshow("fsdf.jpg",img)
#cv2.imshow("asd.jpg",blur)
cv2.waitKey(0)
