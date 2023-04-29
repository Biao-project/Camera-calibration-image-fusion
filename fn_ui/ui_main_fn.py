from fn_ui.ui_pic import Ui_MainWindow, Ui_MainWindow_main
from PyQt5.QtWidgets import QMainWindow

from PyQt5.QtGui import QImage, QPixmap

import cv2 as cv
import json

from fn_image_process.img_calibration import Image_Calibration
from fn_image_process.img_fusion import make_image_fusion


class Window_Function:
    pathA = r'D:\Python\02-job\Camera calibration image fusion\camera_fusion\videoA\A.mp4'
    pathB = r'D:\Python\02-job\Camera calibration image fusion\camera_fusion\videoB\B.mp4'

    fileA = r'./camera_fusion/imagA_calibration.json'
    fileB = r'./camera_fusion/imagB_calibration.json'
    imgs_pathA = r'./camera_fusion/imagesA/*.jpg'
    imgs_pathB = r'./camera_fusion/imagesB/*.jpg'

    def __init__(self, form=None):
        self.Form = QMainWindow()

        self.ui_sign = Ui_MainWindow_main()
        self.ui_sign.setupUi(self.Form)
        self.Form.show()
        self.ui_sign.pushButton.clicked.connect(self.sign_in)
        self.ui_sign.pushButton2.clicked.connect(self.sign_up)

        try:
            self.load_account()
        except:
            self.accout = dict(zhangsan='123456', lisi='123456')
        self.save_account()

        self.debug = True

    def save_account(self):
        with open('./fn_ui/account.json', 'w') as f:
            f.write(json.dumps(self.accout))

    def load_account(self):
        with open('./fn_ui/account.json', 'r') as f:
            self.accout = json.load(f)

    def sign_in(self):
        account = self.ui_sign.textEdit.toPlainText()
        pw = self.ui_sign.textEdit_2.toPlainText()
        sign_success = True
        # print( account, pw , 'exist:', self.accout)
        try:
            if str(self.accout[account]) == pw: sign_success = True
            # print( self.accout[account]  )
        except:
            pass
        cv.waitKey(50)
        if sign_success:
            self.ui_sign.pushButton.setText('success')
            cv.waitKey(800)
            self.open_calibration()
            self.ui_sign.pushButton.setText('Sign In')
            self.Form.close()
        else:
            self.ui_sign.pushButton.setText('Fail!')
            cv.waitKey(1000)
            self.ui_sign.pushButton.setText('Sign In')

    def sign_up(self):
        account = self.ui_sign.textEdit.toPlainText()
        pw = self.ui_sign.textEdit_2.toPlainText()
        sign_success = False
        try:
            self.accout[account] = pw
            # print(self.accout[account])
            sign_success = True
        except:
            pass
        cv.waitKey(50)
        if sign_success:
            self.ui_sign.pushButton2.setText('success')
            cv.waitKey(800)
            self.save_account()
            self.ui_sign.pushButton2.setText('Sign Up')
        else:
            self.ui_sign.pushButton2.setText('Fail!')
            cv.waitKey(1000)
            self.ui_sign.pushButton2.setText('Sign Up')

    """ 打开合成窗口 """

    def open_calibration(self):
        self.ui_img = Ui_MainWindow()
        self.Form2 = QMainWindow()
        self.ui_img.setupUi(self.Form2)
        self.Form2.show()

        self.ui_img.pushButton.clicked.connect(self.image_left_show)
        self.ui_img.pushButton_2.clicked.connect(self.image_right_show)
        self.ui_img.pushButton_3.clicked.connect(self.image_fusion_show)
        self.ui_img.pushButton_5.clicked.connect(self.close_calibration)

    def label_show_img(self, label, img_src=None):
        label_width = label.width()
        label_height = label.height()

        h, w = img_src.shape[:2]
        k = label_width / w
        k2 = label_height / h
        k = min(k, k2)

        img_src = cv.resize(img_src, (0, 0), fx=k, fy=k)
        img_src = cv.cvtColor(img_src, cv.COLOR_BGR2RGB)
        label_width = img_src.shape[1]  # .width()
        label_height = img_src.shape[0]  # .height()

        # 将图片转换为QImage
        temp_imgSrc = QImage(img_src, img_src.shape[1], img_src.shape[0], img_src.shape[1] * 3, QImage.Format_RGB888)
        # 将图片转换为QPixmap方便显示
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)

        # 使用label进行显示
        label.setPixmap(pixmap_imgSrc)

        pass

    """ 关闭合成窗口 """

    def close_calibration(self):
        self.Form2.close()
        self.Form.show()
        del self.ui_img, self.Form2

    def image_left_show(self):
        if self.debug:
            pathA = self.pathA
            pathB = self.pathB
            self.ui_img.textEdit.setText(pathB)
            self.ui_img.textEdit_2.setText(pathA)
        fileA = self.fileA
        fileB = self.fileB
        imgs_pathA = self.imgs_pathA
        imgs_pathB = self.imgs_pathB

        pathA = self.ui_img.textEdit.toPlainText()  # left

        capA = cv.VideoCapture(pathA)

        img_cla = Image_Calibration(fileA, fileB, imgs_pathA, imgs_pathB)

        while True:
            retA, frameA = capA.read()
            if frameA is None: break
            frameA = cv.resize(frameA, (0, 0), fx=0.4, fy=0.4)
            imga = img_cla.calibration(frameA, att='a')

            """ 显示 """
            self.label_show_img(self.ui_img.label, imga)

            cv.waitKey(20)

        capA.release()
        pass

    def image_right_show(self):
        if self.debug:
            pathA = self.pathA
            pathB = self.pathB
            self.ui_img.textEdit.setText(pathB)
            self.ui_img.textEdit_2.setText(pathA)
        fileA = self.fileA
        fileB = self.fileB
        imgs_pathA = self.imgs_pathA
        imgs_pathB = self.imgs_pathB

        pathB = self.ui_img.textEdit_2.toPlainText()  # right

        capB = cv.VideoCapture(pathB)

        img_cla = Image_Calibration(fileA, fileB, imgs_pathA, imgs_pathB)

        # print('===============')
        while True:
            retB, frameB = capB.read()
            if frameB is None: break
            frameB = cv.resize(frameB, (0, 0), fx=0.4, fy=0.4)

            imgb = img_cla.calibration(frameB, att='b')

            """ 显示 """
            self.label_show_img(self.ui_img.label_2, imgb)

            cv.waitKey(20)

        capB.release()

        pass

    def image_fusion_show(self):
        self.calibration()
        pass

    def calibration(self):
        if self.debug:
            pathA = self.pathA
            pathB = self.pathB
            self.ui_img.textEdit.setText(pathB)
            self.ui_img.textEdit_2.setText(pathA)
        fileA = self.fileA
        fileB = self.fileB
        imgs_pathA = self.imgs_pathA
        imgs_pathB = self.imgs_pathB

        pathA = self.ui_img.textEdit.toPlainText()  # left
        pathB = self.ui_img.textEdit_2.toPlainText()  # right

        capA = cv.VideoCapture(pathA)
        capB = cv.VideoCapture(pathB)
        """ 初始化拼接程序，自动加载数据文件 """
        img_cla = Image_Calibration(fileA, fileB, imgs_pathA, imgs_pathB)

        while True:
            retA, frameA = capA.read()
            retB, frameB = capB.read()
            if frameA is None or frameB is None: break
            """ 缩小尺寸，提高速度 """
            frameB = cv.resize(frameB, (0, 0), fx=0.4, fy=0.4)
            frameA = cv.resize(frameA, (0, 0), fx=0.4, fy=0.4)
            """ A图像矫正 """
            imga = img_cla.calibration(frameA, att='a')
            """ B图像矫正 """
            imgb = img_cla.calibration(frameB, att='b')
            """ 图像拼接融合 """
            pano, img = img_cla.calibration_two(image_l=imgb, image_r=imga)
            if img is not None:
                """ 显示 """
                self.label_show_img(self.ui_img.label_3, pano)
            else:
                self.ui_img.label_3.setText('Error!')

            cv.waitKey(1)

        capA.release()
        capB.release()
        # cv.waitKey()
