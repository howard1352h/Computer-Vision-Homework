# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob

# 找棋盤格角點
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盤規格為 11*8
w = 11
h = 8

objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

objpoints = [] 
imgpoints = []

images = glob.glob('*.bmp')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (w,h), corners, ret)

# 返回攝像機矩陣，畸變係數，旋轉和變換向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Draw a “pyramid” on the chessboard images (1.bmp to 5.bmp)
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,0,255),5)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3)
    return img

# Corners  (1, 1, 0)(1, -1, 0)(-1, -1, 0)(-1, 1, 0)
# Vertex   (0, 0, -2)
axis = np.float32([ [1,1,0], [1,-1,0],[-1,-1,0], [-1,1,0],
                   [0,0,-2],[0,0,-2],[0,0,-2],[0,0,-2]])



# Click the button ”2” to show the pyramid on each pictures for 0.5 seconds (total 5 images)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv2.namedWindow("cv2019_hw1",0)
        cv2.imshow('cv2019_hw1',img)
        cv2.waitKey(500) 
cv2.destroyAllWindows()

