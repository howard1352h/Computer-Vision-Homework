# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'layout1.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(691, 412)
        self.label_1 = QtWidgets.QLabel(Form)
        self.label_1.setGeometry(QtCore.QRect(20, 30, 341, 171))
        self.label_1.setStyleSheet("QLabel{border:1px solid rgb(176,176,176);}")
        self.label_1.setText("")
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(390, 30, 281, 311))
        self.label_2.setStyleSheet("QLabel{border:1px solid rgb(176,176,176);}")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(20, 220, 341, 121))
        self.label_3.setStyleSheet("QLabel{border:1px solid rgb(176,176,176);}")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(40, 20, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(40, 210, 241, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label_4.setScaledContents(False)
        self.label_4.setWordWrap(False)
        self.label_4.setOpenExternalLinks(False)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(410, 20, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label_5.setScaledContents(False)
        self.label_5.setWordWrap(False)
        self.label_5.setOpenExternalLinks(False)
        self.label_5.setObjectName("label_5")
        self.Btn_1 = QtWidgets.QPushButton(Form)
        self.Btn_1.setGeometry(QtCore.QRect(40, 60, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.Btn_1.setFont(font)
        self.Btn_1.setStyleSheet("border:2px solid rgb(176,176,176);\n"
"background-color: rgb(211,211,211);")
        self.Btn_1.setObjectName("Btn_1")
        self.Btn_2_1 = QtWidgets.QPushButton(Form)
        self.Btn_2_1.setGeometry(QtCore.QRect(40, 250, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.Btn_2_1.setFont(font)
        self.Btn_2_1.setStyleSheet("border:2px solid rgb(176,176,176);\n"
"background-color: rgb(211,211,211);")
        self.Btn_2_1.setObjectName("Btn_2_1")
        self.Btn_3_1 = QtWidgets.QPushButton(Form)
        self.Btn_3_1.setGeometry(QtCore.QRect(410, 60, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.Btn_3_1.setFont(font)
        self.Btn_3_1.setStyleSheet("border:2px solid rgb(176,176,176);\n"
"background-color: rgb(211,211,211);")
        self.Btn_3_1.setObjectName("Btn_3_1")
        self.Btn_3_2 = QtWidgets.QPushButton(Form)
        self.Btn_3_2.setGeometry(QtCore.QRect(410, 130, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.Btn_3_2.setFont(font)
        self.Btn_3_2.setStyleSheet("border:2px solid rgb(176,176,176);\n"
"background-color: rgb(211,211,211);")
        self.Btn_3_2.setObjectName("Btn_3_2")
        self.Btn_OK = QtWidgets.QPushButton(Form)
        self.Btn_OK.setGeometry(QtCore.QRect(400, 370, 121, 28))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.Btn_OK.setFont(font)
        self.Btn_OK.setObjectName("Btn_OK")
        self.Btn_cancel = QtWidgets.QPushButton(Form)
        self.Btn_cancel.setGeometry(QtCore.QRect(550, 370, 111, 28))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setWeight(50)
        self.Btn_cancel.setFont(font)
        self.Btn_cancel.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.Btn_cancel.setObjectName("Btn_cancel")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        
        self.Btn_1.clicked.connect(self.map1)
        
        self.Btn_2_1.clicked.connect(self.template)
#        
        self.Btn_3_1.clicked.connect(self.keypoints)
        
        self.Btn_3_2.clicked.connect(self.match_line)
    
    def map1(self):
        imgL = cv2.imread('imL.png',0)
        imgR = cv2.imread('imR.png',0)

        stereo = cv2.cv2.StereoBM_create(numDisparities=64, blockSize=9)
        disparity = stereo.compute(imgL,imgR)
        
        norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imshow('L-R Disparity Check',norm_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def template(self):
        img_rgb = cv2.imread('ncc_img.jpg')
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('ncc_template.jpg',0)
        w, h = template.shape[::-1]
        
        # use the Normalized Cross Correlation method
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        for i in np.arange(0.99,1,0.0001):
            min_thresh = max_val*i
            match_locations = np.where(res>=min_thresh)
            num =list(match_locations[0])
            if len(num) == 5:
                weight = i
                break
        #print(weight)

        w, h = template.shape[::-1]
        for (x, y) in zip(match_locations[1], match_locations[0]):
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), [0,0,0], 2)

        norm_res = cv2.normalize(res, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imshow('5 detected template images ', img_rgb)
        cv2.imshow('result of template matching feature', norm_res)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def keypoints(self):
        img1 = cv2.imread('Aerial1.jpg',0)         
        img2 = cv2.imread('Aerial2.jpg',0)
        
        sift = cv2.xfeatures2d.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        def findPoint(kp,des):
            rad = np.asarray([r.size for r in kp])
            rad_sort = np.sort(rad)
            rad_six = rad_sort[-7:]
            rad = list(rad)
            index0 = []
            for i in rad_six:
                index0.append(rad.index(i))
            kp_find = []
            for i in index0:
                kp_find.append(kp[i])
            des_find = []
            for i in index0:
                des_find.append(des[i,:])
            des_find = np.array(des_find)
            des_find = des_find.reshape(7,128)
            return kp_find,des_find
        
        kp1,des1 = findPoint(kp1,des1)
        kp2,des2 = findPoint(kp2,des2)
              
        draw1 = cv2.drawKeypoints(img1,kp1,img1,(0,0,255))
        draw2 = cv2.drawKeypoints(img2,kp2,img2,(0,0,255))
        
        cv2.imshow('FeatureAerial1',draw1)
        cv2.imshow('FeatureAerial2',draw2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite('FeatureAerial1.jpg', draw1)
        cv2.imwrite('FeatureAerial2.jpg', draw2)
    
    def match_line(self):
        img1 = cv2.imread('Aerial1.jpg',0)         
        img2 = cv2.imread('Aerial2.jpg',0)
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        
        
        def findPoint(kp,des):
            rad = np.asarray([r.size for r in kp])
            rad_sort = -np.sort(-rad)
            rad = list(rad)
            sum = 0
            rad_six= []
            rad_six_index = []
            for i in range(len(rad_sort)):
                if rad_sort[i] != rad_sort[i + 1]:
                    sum+=1
                    rad_six.append(rad_sort[i])
                    rad_six_index.append(rad.index(rad_sort[i]))
                    if sum ==6:
                        break
            print(rad_six)
#            print(rad_six_index)
        
            kp_find = []
            for i in rad_six_index:
                kp_find.append(kp[i])
            des_find = []
            for i in rad_six_index:
                des_find.append(des[i,:])
            des_find = np.array(des_find)
            des_find = des_find.reshape(6,128)
            return kp_find,des_find
            
        
        
        kp1,des1 = findPoint(kp1,des1)
        kp2,des2 = findPoint(kp2,des2)
        
        index_params = dict(algorithm = 0, trees = 5) 
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
        
        for j in np.arange(0.6,0.999,0.001):
            matchesMask = [[0,0] for i in range(len(matches))]
            t = 0
            for i,(m,n) in enumerate(matches):
        #        if m.distance <  j*n.distance:
                if m.distance <  200:
                    t+=1
                    matchesMask[i]=[1,0]
                    if t == 4:
                        weight = j
                        break
        
        
        # cv為bgr，matchColor為線的顏色，singlePointColor是點的顏色
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (0,0,255),
                           matchesMask = matchesMask,
                           flags = 0)
        
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        
        cv2.imshow('',img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cv2019_Hw2"))
        self.label.setText(_translate("Form", "1. Stereo"))
        self.label_4.setText(_translate("Form", "2. Normalized Cross Correlation"))
        self.label_5.setText(_translate("Form", "3. SIFT"))
        self.Btn_1.setText(_translate("Form", "1.1 Disparity"))
        self.Btn_2_1.setText(_translate("Form", "2.1 NCC"))
        self.Btn_3_1.setText(_translate("Form", "3.1 Keypoints"))
        self.Btn_3_2.setText(_translate("Form", "3.2 Matched Keypoints"))
        self.Btn_OK.setText(_translate("Form", "OK"))
        self.Btn_cancel.setText(_translate("Form", "Cancel"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication.instance() # checks if QApplication already exists
    if not app: # create QApplication if it doesnt exist
        app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
    


