import cv2 as cv
import numpy as np
import glob
import json
import pickle

from fn_image_process.img_fusion import fn_match_points


def image_calibration(path='./camera_fusion/imagesA/*.jpg', save_file:str=None):
    # save_file = './camera_fusion/imagA_calibration.json'
    CHECKERBOARD = (6, 4)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # cv.TERM_CRITERIA_EPS：精度满足eps时，停止迭代。
    # cv.TERM_CRITERIA_MAX_ITER：迭代次数超过阈值max_iter时，停止迭代。
    # cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER：上述两个条件中的任意一个满足时，停止迭代。

    # 为3D点定义世界坐标
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    # print('objp', objp.shape)
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    # 加载pic文件夹下所有的jpg图像
    images = glob.glob(path)  # 拍摄的十几张棋盘图片所在目录
    gray = None
    i = 0
    for fname in images:
        img = cv.imread(fname)
        # 获取画面中心点
        # 获取图像的长宽
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 找到棋盘格角点
        # ret, corners = cv.findChessboardCorners(gray, (w,h),None)
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        # print('corners:', corners.shape)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            print(f"\rget image: {i}", end='')
            i = i + 1
            # 在原角点的基础上寻找亚像素角点
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 追加进入世界三维点和平面二维点中
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv.namedWindow('findCorners', cv.WINDOW_NORMAL)
            cv.resizeWindow('findCorners', 640, 480)
            cv.imshow('findCorners', img)
            cv.waitKey(2)
    cv.destroyAllWindows()
    print('')
    # %% 标定
    # print('正在计算')
    # 标定
    ret, mtx, dist, rvecs, tvecs = \
        cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print("ret:", ret)
    # print("mtx:\n", mtx)  # 内参数矩阵
    # print("dist畸变值:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecs旋转（向量）外参:\n", rvecs)  # 旋转向量  # 外参数
    # print("tvecs平移（向量）外参:\n", tvecs)  # 平移向量  # 外参数

    if save_file is not None:
        # data = dict(ret=ret, mtx=mtx.tolist(), dist=dist.tolist(), objpoints=np.array(objpoints).tolist(), imgpoints=np.array(imgpoints).tolist())
        data = dict(ret=ret, mtx=mtx, dist=dist, objpoints=objpoints,
                    imgpoints=imgpoints)
        with open(save_file, 'wb') as f:
            # f.write(json.dumps(data, indent=2))
            pickle.dump(data, f)  # 序列化

    return  ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, gray


class Image_Calibration:
    def __init__(self, fileA=r'./camera_fusion/imagA_calibration.json',
                 fileB=r'./camera_fusion/imagB_calibration.json',
                 pathA=r'./camera_fusion/imagesA/*.jpg',
                 pathB=r'./camera_fusion/imagesB/*.jpg',
                 ):
        self.fileA = fileA
        self.fileB = fileB
        self.pathA = pathA
        self.pathB = pathB
        try:
            self.config_a = self.__load_data(self.fileA)
            self.config_b = self.__load_data(self.fileB)
        except:
            image_calibration(pathA, save_file=self.fileA)
            image_calibration(pathB, save_file=self.fileB)

            self.config_a = self.__load_data(self.fileA)
            self.config_b = self.__load_data(self.fileB)


    def __load_data(self, file):
        with open(file, 'rb') as f:
            # data = json.load(f)
            data = pickle.load(f)  # 反序列化
        return data

    def test_calibration(self, pathA=r'./camera_fusion/imagesA/*.jpg',
                 pathB=r'./camera_fusion/imagesB/*.jpg',):
        image_calibration(pathA, save_file=self.fileA)
        image_calibration(pathB, save_file=self.fileB)

    def maximum_internal_rectangle(self, img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY)

        ret, contours, _ = cv.findContours(img_bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        contour = contours[0].reshape(len(contours[0]), 2)

        rect = []

        for i in range(len(contour)):
            x1, y1 = contour[i]
            for j in range(len(contour)):
                x2, y2 = contour[j]
                area = abs(y2 - y1) * abs(x2 - x1)
                rect.append(((x1, y1), (x2, y2), area))

        all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

        if all_rect:
            best_rect_found = False
            index_rect = 0
            nb_rect = len(all_rect)

            while not best_rect_found and index_rect < nb_rect:

                rect = all_rect[index_rect]
                (x1, y1) = rect[0]
                (x2, y2) = rect[1]

                valid_rect = True

                x = min(x1, x2)
                while x < max(x1, x2) + 1 and valid_rect:
                    if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                        valid_rect = False
                    x += 1

                y = min(y1, y2)
                while y < max(y1, y2) + 1 and valid_rect:
                    if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                        valid_rect = False
                    y += 1

                if valid_rect:
                    best_rect_found = True

                index_rect += 1

            if best_rect_found:
                x0, x01 = min(x1, x2), max(x1, x2)
                y0, y01 = min(y1,y2), max(y1, y2)
                img2 = img[y0:y01, x0:x01]

                return img2

            else:
                print("No rectangle fitting into the area")

        else:
            print("No rectangle found")
            return None


    def calibration_two(self, image_l, image_r, att='a', ishow=False):
        h, w = image_l.shape[:2]

        stitcher = cv.createStitcher(False)
        (retval, pano) = stitcher.stitch((image_l, image_r))

        if pano is None: return None, None
        """ 去除拼接后的黑边，并返回无黑边区域的图像 """
        img = self.maximum_internal_rectangle(pano)
        if img is None: return None, None
        h2, w2 = img.shape[:2]
        # print(img.shape)
        img = cv.resize(img, (0,0), fx=h/h2, fy=h/h2)
        # cv.imshow('switch format', img)
        """ 有黑边区域 和 无黑边区域图像返回 """
        return pano, img


    def calibration(self, image, att='a', ishow=False):
        # data = dict(ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        if att.lower() =='a':
            mtx = self.config_a['mtx']
            dist = self.config_a['dist']
        elif att.lower() == 'b':
            mtx = self.config_b['mtx']
            dist = self.config_b['dist']
        else: raise KeyError('att must be "a" or "b"!')
        mtx = np.array(mtx)
        dist = np.array(dist)
        frame = image
        if frame is None: return image
        h, w = frame.shape[:2]

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # # 纠正畸变
        dst1 = cv.undistort(frame, mtx, dist, None, newcameramtx)  # newcameramtx

        """  对图片有效区域进行剪裁"""
        x, y, w1, h1 = roi
        dst1 = dst1[y:y + h1, x:x + w1]
        if ishow:
            cv.imshow('dst1', dst1)
            cv.imshow('or frame', frame)
            cv.waitKey(20)
        dst1 = cv.resize(dst1, ( w, h))
        return dst1



