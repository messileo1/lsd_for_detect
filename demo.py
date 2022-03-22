# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import math
import numpy as np
from PyQt5 import QtCore, QtWidgets

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2, sys, time
from PyQt5.QtWidgets import QApplication, QMainWindow



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1507, 871)
        MainWindow.setStyleSheet("background-color: rgb(99, 148, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(370, -10, 891, 131))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1310, 710, 121, 61))
        self.pushButton.setStyleSheet("font: 22pt \"新宋体\";\n"
"background-color: rgb(255, 0, 0);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1300, 260, 181, 71))
        self.pushButton_2.setStyleSheet("font: 22pt \"新宋体\";\n"
"selection-color: rgb(255, 0, 0);\n"
"background-color: rgb(233, 237, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 90, 1201, 711))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1507, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(self.show_camera)
        self.pushButton.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "动态检测高压线平台（by郑锐）"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:36pt;\">基于视觉的高压线动态检测系统</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "退出"))
        self.pushButton_2.setText(_translate("MainWindow", "动态检测"))

    def show_camera(self):
        self.fileName, self.fileType = QFileDialog.getOpenFileName(self.pushButton, 'Choose file', './video',
                                                                   '*.mp4')
        cap = cv2.VideoCapture(self.fileName)
        self.frameRate = cap.get(cv2.CAP_PROP_FPS)
        # cap = cv2.VideoCapture("rtmp://58.200.131.2:1935/livetv/hunantv")
        ret, img = cap.read()
        time_start = time.time()

        while ret:

            ret, img = cap.read()
            # input = 'images/11.jpg'

            # img = cv2.imread(input, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('img', img)
            lsd = cv2.createLineSegmentDetector(0, _scale=1)

            dlines = lsd.detect(gray)
            k = []
            l = []
            theta = []
            ltheta = np.zeros((8))
            suml = np.zeros((8))
            meanl = np.zeros((8))
            image = img
            image2 = img
            for dline in dlines[0]:
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                # cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
                # cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
                # cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)
                k0 = (dline[0][3] - dline[0][1]) / (dline[0][2] - dline[0][0])
                l0 = ((dline[0][2] - dline[0][0]) ** 2 + (dline[0][3] - dline[0][1]) ** 2) ** 0.5
                theta0 = math.degrees(math.atan(k0))
                # theta0 = int(theta0*0.1)*10

                theta0 = (int(theta0 // 22.5)) * 22.5
                for i in range(0, 8):
                    if theta0 == (i - 4) * 22.5:
                        ltheta[i] += l0
                        suml[i] += 1

                        break
                    else:
                        pass

                k.append(k0)
                l.append(l0)
                theta.append(theta0)
            meanl = ltheta / suml
            where_are_NaNs = np.isnan(meanl)
            meanl[where_are_NaNs] = 0
            # print(k)
            #
            # print(theta)
            # print(l)
            # print(meanl)
            # print(np.argmax(meanl))
            sum = 0
            jieju = []
            x0list = []
            x1list = []
            y0list = []
            y1list = []
            delta = 0  # 判别是否有LSD直线
            for dline in dlines[0]:
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                l0 = ((dline[0][2] - dline[0][0]) ** 2 + (dline[0][3] - dline[0][1]) ** 2) ** 0.5
                theta0 = math.degrees(math.atan((dline[0][3] - dline[0][1]) / (dline[0][2] - dline[0][0])))
                # theta0 = int(theta0 * 0.1) * 10

                if theta0 >= (((np.argmax(meanl)) - 4) * 22.5 - 5) and theta0 <= (
                        ((np.argmax(meanl)) - 3) * 22.5 + 5) and l0 > 180:
                    # if theta0 >= -30 and theta0 <= 0:
                    # cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
                    # cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
                    # cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)
                    sum = sum + 1

                    b = int((x0 * y1 - x1 * y0) / (x0 - x1))  # 直线的截距
                    jieju.append(b)
                    x0list.append(x0)
                    x1list.append(x1)
                    y0list.append(y0)
                    y1list.append(y1)
                    delta = 1
                else:
                    pass
            if delta == 1:
                # print('sum:', sum)
                # print('jieju:', jieju)

                if len(jieju) > 1:
                    X = [[i] for i in jieju]

                    # method是指计算类间距离的方法,比较常用的有3种:
                    # single:最近邻,把类与类间距离最近的作为类间距
                    # average:平均距离,类与类间所有pairs距离的平均
                    # complete:最远邻,把类与类间距离最远的作为类间距
                    Z = linkage(X, 'single')
                    f = fcluster(Z, 10, 'distance')
                    # print(f)
                    # fig = plt.figure(figsize=(5, 3))
                    # dn = dendrogram(Z)
                    # plt.show()

                    for i in range(1, f.max() + 1):
                        xzanshi = []
                        yzanshi = []

                        for j in range(len(jieju)):
                            if f[j] == i:
                                xzanshi.append(x0list[j])
                                yzanshi.append(y0list[j])
                                xzanshi.append(x1list[j])
                                yzanshi.append(y1list[j])
                        # print(xzanshi)
                        # print(yzanshi)
                        k = np.polyfit(xzanshi, yzanshi, 1)
                        # print('k=', k)
                        cv2.line(image2, (0, int(k[1])), (1500, int(1500 * k[0] + k[1])), (0,0,255), 1, cv2.LINE_AA)
                else:
                    pass

            # cv2.imshow('new', image2)

            # RGB转BGR
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            image2 = QImage(image2.data, image2.shape[1], image2.shape[0], QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(image2))

            time_end = time.time()
            if time_end - time_start > 9:
                self.label_2.clear()
                break
            else:
                cv2.waitKey(20)

        # self.label_2.clear()

        # success, frame = cap.read()
        # self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # # x, y = frame.shape[0:2]
        # # self.frame = cv2.resize(self.frame, (int(3 * y / 4), int(3 * x / 4)))
        # img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
        # self.label_2.setPixmap(QPixmap.fromImage(img))
        # cv2.waitKey(40)

    # 创造一个线程
    # def Open(self):
    #     th = threading.Thread(target=self.show_camera)
    #     th.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())