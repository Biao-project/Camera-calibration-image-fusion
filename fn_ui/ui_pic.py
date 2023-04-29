# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designerZNmBRt.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 629)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 381, 281))
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(300, 13, 75, 41))
        font = QFont()
        font.setFamily(u"Times New Roman")
        font.setPointSize(12)
        self.pushButton.setFont(font)

        self.textEdit = QTextEdit(self.groupBox)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(10, 20, 281, 30))
        font1 = QFont()
        font1.setFamily(u"Times New Roman")
        font1.setPointSize(10)
        self.textEdit.setFont(font1)
        self.groupBox_2 = QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 60, 361, 211))
        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(4, 0, 350, 210))
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(410, 10, 381, 281))
        self.pushButton_2 = QPushButton(self.groupBox_3)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(300, 13, 75, 41))
        self.pushButton_2.setFont(font)
        self.textEdit_2 = QTextEdit(self.groupBox_3)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(10, 20, 281, 30))
        self.textEdit_2.setFont(font1)
        self.groupBox_4 = QGroupBox(self.groupBox_3)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(10, 60, 361, 211))
        self.label_2 = QLabel(self.groupBox_4)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(4, 0, 350, 210))
        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(10, 300, 781, 281))
        self.pushButton_3 = QPushButton(self.groupBox_5)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(700, 20, 75, 41))
        self.pushButton_3.setFont(font)
        self.groupBox_6 = QGroupBox(self.groupBox_5)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setGeometry(QRect(10, 20, 681, 251))
        self.label_3 = QLabel(self.groupBox_6)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(4, 0, 671, 210))
        self.pushButton_4 = QPushButton(self.groupBox_5)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(700, 180, 75, 41))
        self.pushButton_4.setFont(font)
        self.pushButton_5 = QPushButton(self.groupBox_5)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(700, 230, 75, 41))
        self.pushButton_5.setFont(font)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"camera left", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.groupBox_2.setTitle("")
        self.label.setText("")
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"camera right", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.groupBox_4.setTitle("")
        self.label_2.setText("")
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"camera calibration", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.groupBox_6.setTitle("")
        self.label_3.setText("")
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"\u9884\u7559", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Back", None))
    # retranslateUi



class Ui_MainWindow_main(object):

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(110, 100, 551, 401))
        font = QFont()
        font.setFamily(u"Adobe Devanagari")
        self.groupBox.setFont(font)
        self.textEdit = QTextEdit(self.groupBox)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(150, 70, 331, 31))
        font1 = QFont()
        font1.setFamily(u"Times New Roman")
        font1.setPointSize(12)
        self.textEdit.setFont(font1)
        self.textEdit.viewport().setProperty("cursor", QCursor(Qt.IBeamCursor))
        self.textEdit.setMouseTracking(False)
        self.textEdit.setTabletTracking(True)
        self.textEdit.setFocusPolicy(Qt.ClickFocus)
        self.textEdit.setAcceptDrops(False)
        self.textEdit.setAutoFillBackground(False)
        self.textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(50, 60, 91, 41))
        font2 = QFont()
        font2.setFamily(u"Times New Roman")
        font2.setPointSize(14)
        self.label.setFont(font2)
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(50, 130, 91, 41))
        self.label_2.setFont(font2)
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(40, 210, 471, 61))
        font3 = QFont()
        font3.setFamily(u"Times New Roman")
        font3.setPointSize(16)
        self.pushButton.setFont(font3)

        self.pushButton2 = QPushButton(self.groupBox)
        self.pushButton2.setObjectName(u"pushButton")
        self.pushButton2.setGeometry(QRect(40, 310, 471, 61))
        self.pushButton2.setFont(font3)

        self.textEdit_2 = QTextEdit(self.groupBox)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(150, 140, 331, 31))
        self.textEdit_2.setFont(font1)
        self.textEdit_2.viewport().setProperty("cursor", QCursor(Qt.IBeamCursor))
        self.textEdit_2.setMouseTracking(False)
        self.textEdit_2.setTabletTracking(True)
        self.textEdit_2.setFocusPolicy(Qt.ClickFocus)
        self.textEdit_2.setAcceptDrops(False)
        self.textEdit_2.setAutoFillBackground(False)
        self.textEdit_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Account sign in.", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Account\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Password\uff1a", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Sign In", None))
        self.pushButton2.setText(QCoreApplication.translate("MainWindow", u"Sign Up", None))
    # retranslateUi


if __name__ == '__main__':
    import sys
    # from MainWidget import MainWidget
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWidgets import QMainWindow

    app = QApplication(sys.argv)

    # Form = MainWidget()  # 新建一个主界面
    # Form.show()  # 显示主界面

    Form = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
