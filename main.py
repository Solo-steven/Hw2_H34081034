from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torchsummary import torchsummary
from torchvision import models

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("HW2")
        MainWindow.resize(1212, 463)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        """
        File Path
        """
        self.base_path = "./"
        self.label_file_path = QtWidgets.QLabel(self.centralwidget)
        self.label_file_path.setGeometry(QtCore.QRect(10, 10, 150, 20))
        self.label_file_path.setText(_translate("MainWindow", "File Path"));
        self.label_file_path.setObjectName("label_file_path")
        self.frame_file_path = QtWidgets.QFrame(self.centralwidget)
        self.frame_file_path.setGeometry(QtCore.QRect(10, 20, 200, 200))
        self.frame_file_path.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_file_path.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_file_path.setObjectName("frame_file_path")
        self.input_file_path = QtWidgets.QLineEdit(self.frame_file_path);
        self.input_file_path.setGeometry(QtCore.QRect(20, 20, 150, 20))
        self.button_file_path = QtWidgets.QPushButton(self.frame_file_path);
        self.button_file_path.setObjectName("button_file_path")
        self.button_file_path.setGeometry(QtCore.QRect(20, 50, 150, 30))
        self.button_file_path.setText(_translate("MainWindow", "Update File Path"))
        """
        Qustion 1
        """
        self.label_q1 = QtWidgets.QLabel(self.centralwidget)
        self.label_q1.setGeometry(QtCore.QRect(260, 10, 180, 20))
        self.label_q1.setObjectName("label_q1")
        self.label_q1.setText(_translate("MainWindow", "1. Hough Circle Transform"))
        self.frame_q1 = QtWidgets.QFrame(self.centralwidget)
        self.frame_q1.setGeometry(QtCore.QRect(260, 20, 221, 170))
        self.frame_q1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_q1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_q1.setObjectName("frame_q1")

        self.button_q1_1 = QtWidgets.QPushButton(self.frame_q1)
        self.button_q1_1.setGeometry(QtCore.QRect(20, 20, 150, 40))
        self.button_q1_1.setObjectName("button_q1_1")
        self.button_q1_1.setText(_translate("MainWindow", "Draw Contour"))
        self.button_q1_1.clicked.connect(self.solve_q1_1)

        self.button_q1_2 = QtWidgets.QPushButton(self.frame_q1)
        self.button_q1_2.setGeometry(QtCore.QRect(20, 60, 150, 40))
        self.button_q1_2.setObjectName("button_q1_2")
        self.button_q1_2.setText(_translate("MainWindow", "Count Rings"))
        self.button_q1_2.clicked.connect(self.solve_q1_2)
        self.label_q1_2 = QtWidgets.QLabel(self.frame_q1)
        self.label_q1_2.setGeometry(QtCore.QRect(20, 110, 150, 50))
        """
        Qustion 2
        """
        self.label_q2 = QtWidgets.QLabel(self.centralwidget)
        self.label_q2.setGeometry(QtCore.QRect(260, 230, 180, 20))
        self.label_q2.setObjectName("label_q2")
        self.label_q2.setText(_translate("MainWindow", "2. Histogram Equalization"))
        self.frame_q2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_q2.setGeometry(QtCore.QRect(260, 230, 221, 170))
        self.frame_q2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_q2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_q2.setObjectName("frame_q2")

        self.button_q2 = QtWidgets.QPushButton(self.frame_q2)
        self.button_q2.setGeometry(QtCore.QRect(20, 20, 150, 40))
        self.button_q2.setObjectName("button_q2")
        self.button_q2.setText(_translate("MainWindow", "Histogram Equalization"))
        self.button_q2.clicked.connect(self.solve_q2)
        """
        Qustion 3
        """
        self.label_q3 = QtWidgets.QLabel(self.centralwidget)
        self.label_q3.setGeometry(QtCore.QRect(500, 10, 180, 20))
        self.label_q3.setObjectName("label_q3")
        self.label_q3.setText(_translate("MainWindow", "3. Morphology Operation"))
        self.frame_q3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_q3.setGeometry(QtCore.QRect(500, 20, 221, 170))
        self.frame_q3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_q3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_q3.setObjectName("frame_q3")

        self.button_q3_1 = QtWidgets.QPushButton(self.frame_q3)
        self.button_q3_1.setGeometry(QtCore.QRect(20, 20, 150, 40))
        self.button_q3_1.setObjectName("button_q3_1")
        self.button_q3_1.setText(_translate("MainWindow", "Closing"))
        self.button_q3_1.clicked.connect(self.solve_q3_1)

        self.button_q3_2 = QtWidgets.QPushButton(self.frame_q3)
        self.button_q3_2.setGeometry(QtCore.QRect(20, 60, 150, 40))
        self.button_q3_2.setObjectName("button_q3_2")
        self.button_q3_2.setText(_translate("MainWindow", "Opening"))
        self.button_q3_2.clicked.connect(self.solve_q3_2)
        """
        Qustion 4
        """
        self.label_q4 = QtWidgets.QLabel(self.centralwidget)
        self.label_q4.setGeometry(QtCore.QRect(750, 10, 300, 20))
        self.label_q4.setObjectName("label_q4")
        self.label_q4.setText(_translate("MainWindow", "4. Training a MNIST Classifier Using VGG19 "))
        self.frame_q4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_q4.setGeometry(QtCore.QRect(750, 20, 300, 170))
        self.frame_q4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_q4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_q4.setObjectName("frame_q4")

        self.button_q4_1 = QtWidgets.QPushButton(self.frame_q4)
        self.button_q4_1.setGeometry(QtCore.QRect(20, 20, 150, 40))
        self.button_q4_1.setObjectName("button_q4_1")
        self.button_q4_1.setText(_translate("MainWindow", "Show struct"))
        self.button_q4_1.clicked.connect(self.solve_q4_1)

        self.label_file_path.raise_()
        self.frame_file_path.raise_()
        self.frame_q1.raise_()
        self.label_q1.raise_()
        self.frame_q2.raise_();
        self.label_q2.raise_();
        self.label_q3.raise_();
        self.frame_q3.raise_();
        self.label_q4.raise_();
        self.frame_q4.raise_();
        # self.label_2.raise_()
        # self.frame_2.raise_()
        # self.label_3.raise_()
        # self.frame_3.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1012, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # self.label_3.setText(_translate("MainWindow", "3. Edge Detection"))

    def read_image(self, path: str):
        return cv2.imread(self.base_path + path);

    def show_image(self, title, pic):
        msg = QtWidgets.QMessageBox()
        h,w,_ = pic.shape
        bytesPerline = 3 * w
        msg.setWindowTitle(title)
        msg_image = QImage(pic, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        msg.setIconPixmap(QPixmap.fromImage(msg_image))
        msg.exec()
    
    def solve_q1_1(self):
        image = self.read_image("dataset/Q1/coins.jpg")
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=30, maxRadius=40, minRadius=20)
        circles = np.uint16(np.around(circles))

        original_image = self.read_image("dataset/Q1/coins.jpg")
        self.show_image("original", original_image)
        index = 0;
        for i in circles[0,:]:
            index +=1
            cv2.circle(original_image, (i[0],i[1]), i[2], (0, 255, 0),2)
            cv2.circle(original_image, (i[0],i[1]), 2, (0, 0, 255),3)
        self.show_image("original", original_image)

    def solve_q1_2(self):
        count = 0
        image = self.read_image("dataset/Q1/coins.jpg")
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=30, maxRadius=40, minRadius=20)
        circles = np.uint16(np.around(circles))
        for _ in circles[0,:]:
            count +=1
        _translate = QtCore.QCoreApplication.translate
        self.label_q1_2.setText(_translate("MainWindow", "There Are Total {} Coin".format(count)))
    
    def solve_q2(self):
        image = self.read_image("dataset/Q2/histoEqualGray2.png")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2_image = np.uint16(np.around(cv2.equalizeHist(image)));
        
        hist,_ = np.histogram(image.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        manual_image = cdf[image]

        self.show_image("origin", image)
        self.show_image("cv2", cv2_image)
        self.show_image("maunual", manual_image)

        image_range = self.get_range_of_gray(image)
        cv2_image_range = self.get_range_of_gray(cv2_image)
        manual_image_range = self.get_range_of_gray(manual_image)
        self.plt_bar(image_range, cv2_image_range, manual_image_range)


    def get_range_of_gray(self, image):
        image_rangs = [0] * 256;
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                 if image[i][j] < 0 or image[i][j] > 256:
                    print("{}, {}, {}".format(image[i][j]-1, i, j))
                 image_rangs[image[i][j]] += 1;
        return image_rangs
    
    def plt_bar(self, image_range, cv2_image_range, manual_image_range):
        x = []
        for i in range(0, 256):
            x.append(i)
        plt.subplot(1,3,1)
        plt.bar(x, image_range)
        plt.subplot(1,3,2)
        plt.bar(x, cv2_image_range)
        plt.subplot(1,3,3)
        plt.bar(x, manual_image_range)
        plt.show()

    def solve_q3_1(self):
        image = self.read_image("dataset/Q3/closing.png");
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernal = np.ones((3,3))
        image = self.erosion(self.dilation(image_gray, kernal), kernal)
        self.show_image("closing", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    def solve_q3_2(self):
        image = self.read_image("dataset/Q3/opening.png");
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernal = np.ones((3,3))
        image = self.dilation(self.erosion(image_gray, kernal), kernal)
        self.show_image("opening", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    def erosion(self, image, kernal):
        m,n= image.shape 
        k = kernal.shape[0]
        constant= (k-1)//2
        imgErode= np.zeros((m,n), dtype=np.uint8)
        for i in range(constant, m-constant):
            for j in range(constant,n-constant):
                temp= image[i-constant:i+constant+1, j-constant:j+constant+1]
                product= temp*kernal
                imgErode[i,j]= np.min(product)
        return imgErode;

    def dilation(self, image, kernal):
        p,q= image.shape 
        constant1 = 1
        imgDilate= np.zeros((p,q), dtype=np.uint8)
        for i in range(constant1, p-constant1):
            for j in range(constant1,q-constant1):
                temp= image[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
                product= temp* kernal
                imgDilate[i,j]= np.max(product)
        return imgDilate;

    def solve_q4_1(self):
        net =  ModifiedVGG19();
        torchsummary.summary(net, (3, 244, 244))
    
    def solve_q4_2(self):
        pass

class ModifiedVGG19(nn.Module):
    def __init__(self):
        super(ModifiedVGG19, self).__init__()

        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19_features = list(self.vgg19.features.children())

        self.modified_features = [nn.BatchNorm2d(3)]  
        self.modified_features.extend(self.vgg19_features)

        self.vgg19.features = nn.Sequential(*self.modified_features)

    def forward(self, x):
        return self.vgg19(x)

if __name__ == "__main__":
    net = ModifiedVGG19();
    torchsummary.summary(net, (3, 244, 244))
    import sys
    # app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    # sys.exit(app.exec_())
