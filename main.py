import cv2 as cv
import numpy as np
from fn_image_process.img_calibration import Image_Calibration
from fn_image_process.img_fusion import make_image_fusion


def calibration():
    """ 调试程序专用（不运行软件界面，可直接查看报错信息和位置）"""
    pathA = r'D:\Python\02-job\Camera calibration image fusion\camera_fusion\videoA\A.mp4'
    pathB = r'D:\Python\02-job\Camera calibration image fusion\camera_fusion\videoB\B.mp4'

    fileA = r'./camera_fusion/imagA_calibration.json'
    fileB = r'./camera_fusion/imagB_calibration.json'
    imgs_pathA = r'./camera_fusion/imagesA/*.jpg'
    imgs_pathB = r'./camera_fusion/imagesB/*.jpg'

    capA = cv.VideoCapture(pathA)
    retA, frameA = capA.read()
    capB = cv.VideoCapture(pathB)
    retB, frameB = capB.read()
    print('image size:', frameB.shape)
    # cv.imshow('dd', frameB)
    # cv.waitKey()

    """ 创建合成对象：初始化数据 """
    img_cla = Image_Calibration(fileA, fileB, imgs_pathA, imgs_pathB)

    print('===============')
    while True:
        retA, frameA = capA.read()
        retB, frameB = capB.read()
        if frameA is None or frameB is None: break
        frameB = cv.resize(frameB, (0, 0), fx=0.4, fy=0.4)
        frameA = cv.resize(frameA, (0, 0), fx=0.4, fy=0.4)
        """ 拼接函数 """
        pano, img = img_cla.calibration_two(image_l=frameA, image_r=frameB)
        cv.waitKey(200)

    capA.release()
    capB.release()
    cv.waitKey()


def app_start():
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    from fn_ui.ui_main_fn import Window_Function

    win = Window_Function()
    sys.exit(app.exec_())


if __name__ == '__main__':

    is_debug = False

    if is_debug:
        """ 调试程序专用（不运行软件界面，可直接查看拼接程序报错信息和位置）"""
        calibration()
    else:
        """ 运行软件界面和功能  """
        app_start()

    pass
