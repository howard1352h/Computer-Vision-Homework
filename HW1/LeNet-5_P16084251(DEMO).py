# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test2.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import sys
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        
train_images, test_images = train_images / 255.0, test_images / 255.0
            
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

new_model = keras.models.load_model('model_fifty_epochs.h5')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.oneBtn = QtWidgets.QPushButton(self.centralwidget)
        self.oneBtn.setGeometry(QtCore.QRect(40, 70, 211, 31))
        self.oneBtn.setObjectName("oneBtn")
        self.twoBtn = QtWidgets.QPushButton(self.centralwidget)
        self.twoBtn.setGeometry(QtCore.QRect(40, 120, 211, 31))
        self.twoBtn.setObjectName("twoBtn")
        self.threeBtn = QtWidgets.QPushButton(self.centralwidget)
        self.threeBtn.setGeometry(QtCore.QRect(40, 170, 211, 31))
        self.threeBtn.setObjectName("threeBtn")
        self.fourBtn = QtWidgets.QPushButton(self.centralwidget)
        self.fourBtn.setGeometry(QtCore.QRect(40, 220, 211, 31))
        self.fourBtn.setObjectName("fourBtn")
        self.fiveBtn = QtWidgets.QPushButton(self.centralwidget)
        self.fiveBtn.setGeometry(QtCore.QRect(50, 330, 211, 31))
        self.fiveBtn.setObjectName("fiveBtn")
        self.indexShow_label = QtWidgets.QLabel(self.centralwidget)
        self.indexShow_label.setGeometry(QtCore.QRect(20, 270, 101, 31))
        self.indexShow_label.setObjectName("indexShow_label")
        self.parameterLbl = QtWidgets.QLabel(self.centralwidget)
        self.parameterLbl.setGeometry(QtCore.QRect(310, 50, 221, 121))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.parameterLbl.setFont(font)
        self.parameterLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.parameterLbl.setText("")
        self.parameterLbl.setScaledContents(False)
        self.parameterLbl.setObjectName("parameterLbl")
        self.ImageLbl = QtWidgets.QLabel(self.centralwidget)
        self.ImageLbl.setGeometry(QtCore.QRect(310, 210, 700, 600))
        self.ImageLbl.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ImageLbl.setText("")
        self.ImageLbl.setObjectName("ImageLbl")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(130, 280, 141, 22))
        self.lineEdit.setInputMask("")
        self.lineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(50, 290, 101, 31))
        self.label_5.setObjectName("label_5")
        self.showLbl = QtWidgets.QLabel(self.centralwidget)
        self.showLbl.setGeometry(QtCore.QRect(560, 50, 221, 121))
        self.showLbl.setText("")
        self.showLbl.setObjectName("showLbl")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 896, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # 自己新增的觸發
        self.oneBtn.clicked.connect(self.showImage)
        
        self.twoBtn.clicked.connect(self.showParameters)
        
        self.threeBtn.clicked.connect(self.one_epoch)
        
        self.fourBtn.clicked.connect(self.result50_show)
        
        self.fiveBtn.clicked.connect(self.prediction_show)
        
    # 5.1 Load Cifar-10 training dataset and randomly show 10 images and labels respectively    
    def showImage(self):
        import random
        plt.figure(figsize=(10,10))     
        for i in range(10):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            x = random.randint(0,9000)
            plt.imshow(train_images[x], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[x][0]])
        plt.savefig('img_show.png')
        pixmap = QtGui.QPixmap('img_show.png') 
        pixmap = pixmap.scaled(self.ImageLbl.width(), self.ImageLbl.height(), QtCore.Qt.KeepAspectRatio) 
        self.ImageLbl.setPixmap(pixmap) 
        self.ImageLbl.setAlignment(QtCore.Qt.AlignCenter) 
    
    # 5.2 Print out training hyperparameters (batch size, learning rate, optimizer)
    def showParameters(self):
        self.parameterLbl.setText("hyperparameterss:\nbatch_size: 32\nlearning rate: 0.001\noptimizer: SGD") 
        print("hyperparameterss:\nbatch_size: 32\nlearning rate: 0.001\noptimizer: SGD\n")
    
    # 5.3 Train 1 epoch from initial status and show training loss at the end of the epoch
    def one_epoch(self):
        from keras.callbacks import Callback
        from tensorflow.keras import layers, models
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        
        adag = keras.optimizers.SGD(lr= 0.001)
        model.compile(optimizer= adag,
                      validation_split=0.2,
                      batch_size =32,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
        class Histories(Callback):
            def on_train_begin(self,logs={}):
                self.losses = []
        
            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
                
        histories = Histories()
        history = model.fit(train_images, train_labels, epochs = 1, verbose = 1,callbacks=[histories])
        
        interation = list(range(0,len(histories.losses)))
        plt.figure()
        plt.plot(interation,histories.losses)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('interation')
        plt.legend(['Train'], loc='upper right')
        plt.savefig('one_epoch.png')
        
        pixmap = QtGui.QPixmap('one_epoch.png') 
        pixmap = pixmap.scaled(self.ImageLbl.width(), self.ImageLbl.height(), QtCore.Qt.KeepAspectRatio)
        self.ImageLbl.setPixmap(pixmap) 
        self.ImageLbl.setAlignment(QtCore.Qt.AlignCenter) 
     
    # 5.4 Show the screenshot of your training loss and accuracy for 50 epochs 
    def result50_show(self):       
        pixmap = QtGui.QPixmap('train_fifty_epochs.png') 
        pixmap = pixmap.scaled(self.ImageLbl.width(), self.ImageLbl.height(), QtCore.Qt.KeepAspectRatio) 
        self.ImageLbl.setPixmap(pixmap) 
        self.ImageLbl.setAlignment(QtCore.Qt.AlignCenter) 
        self.lineEdit.clear() 
    
    # 5.5 Load your model trained at 5.4.  Let us choose one test image from Cifar-10 test images. 
    # Then inference the image, show image and estimate this test image
    def prediction_show(self):
        predictions = new_model.predict(test_images)
        value = self.lineEdit.text()
        value = int(value)
        plt.figure()
        plt.imshow(test_images[value])
        plt.savefig('show.png')
        plt.figure(figsize=(10,6))
        plt.bar(class_names,predictions[value])
        plt.savefig('prediction.png')
        pixmap = QtGui.QPixmap('prediction.png') 
        pixmap = pixmap.scaled(self.ImageLbl.width(), self.ImageLbl.height(), QtCore.Qt.KeepAspectRatio) 
        self.ImageLbl.setPixmap(pixmap) 
        self.ImageLbl.setAlignment(QtCore.Qt.AlignCenter) 
        
        pixmap = QtGui.QPixmap('show.png')
        pixmap = pixmap.scaled(self.showLbl.width(), self.showLbl.height(), QtCore.Qt.KeepAspectRatio)
        self.showLbl.setPixmap(pixmap) 
        self.showLbl.setAlignment(QtCore.Qt.AlignCenter) 
        self.lineEdit.clear() 
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.oneBtn.setText(_translate("MainWindow", "5.1 Show Train Images"))
        self.twoBtn.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.threeBtn.setText(_translate("MainWindow", "5.3 Train 1 epoch"))
        self.fourBtn.setText(_translate("MainWindow", "5.4 Show Training Result"))
        self.fiveBtn.setText(_translate("MainWindow", "5.5 Inference"))
        self.indexShow_label.setText(_translate("MainWindow", "Test Image Index:"))
        self.label_5.setText(_translate("MainWindow", "(0~9999)"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

