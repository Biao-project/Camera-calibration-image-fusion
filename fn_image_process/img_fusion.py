import cv2 as cv
import numpy as np


def fn_match_points(query_img, train_img, is_automatch=True, ishow=True):
    target = query_img
    board = train_img
    # cv.imshow(';', target)
    target = cv.blur(target, (3, 3))
    board = cv.blur(board, (3, 3))

    # 构造生成器
    sift = cv.xfeatures2d.SIFT_create()
    # 检测图片
    kp1, des1 = sift.detectAndCompute(target, None)  # 关键点（Keypoint）和描述子（Descriptor）
    kp2, des2 = sift.detectAndCompute(board, None)

    # if ishow: # 绘出关键点
    #     target_draw = cv.drawKeypoints(target, kp1, None, (255, 0, 0), 4)
    #     board_draw = cv.drawKeypoints(board, kp2, None, (255, 0, 0), 4)
    #     cv.imshow('query_img', target_draw)  # 打印图像
    #     # cv.imshow('train_img', board_draw)  # 打印图像

    """
    keypoint是检测到的特征点的列表
    descriptor是检测到特征的局部图像的列表
    """
    # 获取flann匹配器
    FLANN_INDEX_KDTREE = 0
    # 参数1：indexParams
    #    对于SIFT和SURF，可以传入参数index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)。
    #    对于ORB，可以传入参数index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12）。
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # 参数2：searchParams 指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多。
    searchParams = dict(checks=50)

    # 使用FlannBasedMatcher 寻找最近邻近似匹配
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    # 使用knnMatch匹配处理，并返回匹配matches
    matches = flann.knnMatch(des1, des2, k=2)

    query_pts = []
    train_pts = []
    thred_value = 0.1

    def auto_match():
        if not ishow:
            # 通过掩码方式计算有用的点
            # 通过描述符的距离进行选择需要的点
            for i, (m, n) in enumerate(matches):
                if m.distance < thred_value * n.distance:  # 通过0.7系数来决定匹配的有效关键点数量
                    query_pts.append(kp1[m.queryIdx].pt)  # [x,y]
                    train_pts.append(kp2[m.trainIdx].pt)  # [x,y]

        else:
            # 通过掩码方式计算有用的点
            matchesMask = [[0, 0] for i in range(len(matches))]
            # 通过描述符的距离进行选择需要的点
            for i, (m, n) in enumerate(matches):
                if m.distance < thred_value * n.distance:  # 通过0.7系数来决定匹配的有效关键点数量
                    matchesMask[i] = [1, 0]
                    query_pts.append(kp1[m.queryIdx].pt)  # [x,y]
                    train_pts.append(kp2[m.trainIdx].pt)  # [x,y]
                    # if len(query_pts)>4: break

            drawPrams = dict(matchColor=(0, 255, 0),
                             singlePointColor=(255, 0, 0),
                             matchesMask=matchesMask,
                             flags=0)
            # 匹配结果图片
            img33 = cv.drawMatchesKnn(target, kp1, board, kp2, matches, None, **drawPrams)
            cv.imshow('match', img33)  # 打印图像

    while True:
        # if ishow: print('current thred_value:', thred_value)
        auto_match()
        if len(query_pts) >= 10 or not is_automatch: break
        thred_value += 0.01

    if len(query_pts) == 0 or len(train_pts) == 0:
        if ishow:
            print('\033[31m find no pt match!\033[0m')
            cv.waitKey(2000)
        return None, None
    return query_pts, train_pts


def get_rect_area(pts: np.array):
    min_ps = np.min(pts, axis=0)
    max_ps = np.max(pts, axis=0)
    # print(min_ps, max_ps)
    return min_ps, max_ps


class Choice_Area:
    def __init__(self, grid=3, depart=2):
        self.index_grid = np.zeros(grid ** 2, dtype=np.bool_)
        self.grid = grid
        self.depart = depart
        self.selected = None
        pass

    def __format_area_index(self, idx):
        x = idx % self.grid
        y = idx // self.grid
        return np.array([x, y])
        pass

    def random_choice(self, num=1):
        """ 随机选择一块区域，确保与已选的区域间隔保持在 self.depart 以上 """
        if self.selected is None:
            temp = np.arange(self.grid ** 2)
            choice = np.random.choice(temp, 1)[0]
            self.selected = np.array([choice])
            return self.__format_area_index(choice)
        else:
            temp = []
            for i in range(self.grid ** 2):
                # print(i, '===',  (i - self.selected), (i - self.selected).min() )
                if abs(i - self.selected).min() >= self.depart: temp.append(i)
            if len(temp) < 1: raise ValueError('image area random_choice had been broken!')
            choice = np.random.choice(temp, 1)[0]
            self.selected = np.hstack([self.selected, choice])
            return self.__format_area_index(choice)

    def split_area(self, image, min_ps, max_ps):
        self.image, self.min_ps, self.max_ps = image, min_ps, max_ps
        pass

    def get_image_area(self):
        image, min_ps, max_ps = self.image, self.min_ps, self.max_ps
        step_num = self.grid
        area = max_ps - min_ps  # [x,y]
        step_area = area / step_num
        area_idx = self.random_choice()  # [x,y]
        print('\tarea_idx:', area_idx, min_ps, area, step_area)
        new_area_start = area_idx * step_area + min_ps  # [x,y]
        new_area_end = (area_idx + 1) * step_area + min_ps  # [x,y]
        new_area_start = np.array(new_area_start, dtype=np.int32)  # [x,y]
        new_area_end = np.array(new_area_end, dtype=np.int32)
        xs, ys, xe, ye = new_area_start[0], new_area_start[1], new_area_end[0], new_area_end[1]
        # print(self.selected, xs, ys, xe, ye, image.shape)
        img = image[ys:ye, xs:xe]
        return img.copy()


""" 定位两个图片的重复区域 """


# query_pts, train_pts = fn_match_points(query_img, train_img, ishow=True)
#
# # cv.waitKey()
#
# if query_pts:
#     query_pts, train_pts = np.array(query_pts, dtype=np.int32), np.array(train_pts, dtype=np.int32)
#
# query_min_ps, query_max_ps = get_rect_area(query_pts)
# # train_min_ps, train_max_ps = get_rect_area(train_pts)
#
# area = Choice_Area(5, 2)
# area.split_area(frameA, query_min_ps, query_max_ps)
# i=0
# while i< 3:
#     img = area.get_image_area()
#     query_pts, train_pts = fn_match_points(img, train_img, ishow=True)
#     if query_pts is None:
#         continue
#     i +=1
#     print('query_pts len=', len(query_pts))
#     query_pts = np.array(query_pts, dtype=np.int32)
#     mean_pt = query_pts.mean(axis=0)
#     print(' mean_py=', mean_pt)
#
#     cv.waitKey(2000)

def make_image_fusion(train_img_left, query_img_right):
    """ 定位两个图片的重复区域 """
    high, width = train_img_left.shape[:2]
    query_img, train_img = query_img_right, train_img_left
    query_pts, train_pts = fn_match_points(query_img.copy(), train_img.copy(), is_automatch=True, ishow=True)

    # src_pts = np.float32(query_pts)
    # dst_pts = np.float32(train_pts)
    # M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    # print(M)
    # train_img=result = cv.warpPerspective(train_img, M, (query_img.shape[1]+train_img.shape[1] , query_img.shape[0]))
    # cv.imshow('aaa', result)
    #
    # # 计算变换后的四个顶点坐标位置
    # train_pts = cv.perspectiveTransform(dst_pts, M)


    query_pts = np.array(query_pts, dtype=np.int32)
    train_pts = np.array(train_pts, dtype=np.int32)
    query_mean_pt = query_pts.mean(axis=0).astype(np.int32)
    train_mean_pt = train_pts.mean(axis=0).astype(np.int32)
    # print('匹配点为：', query_mean_pt, train_mean_pt)
    qx, qy, tx, ty = query_mean_pt[0], query_mean_pt[1], train_mean_pt[0], train_mean_pt[1]

    imga, imgb = query_img[:, qx:], train_img[:, :tx]
    # print('need fusion image shope=', imga.shape, imgb.shape)
    img_fusion = np.concatenate([imgb, imga], axis=1)

    if ty - qy > 0:
        # print('ty>qy')
        img_fusion[:-(ty - qy), :tx] = img_fusion[(ty - qy):, :tx]
        img_fusion[-(ty - qy):, :] = 0
        img_fusion = img_fusion[:-(ty - qy), :]
    elif ty - qy < 0:
        # print('ty<qy')
        img_fusion[-(ty - qy):, :tx] = img_fusion[:(ty - qy), :tx]
        img_fusion[:-(ty - qy), :] = 0
        img_fusion = img_fusion[-(ty - qy):, :]

    img_fusion = cv.resize(img_fusion, (width, high))

    # print(img_fusion.shape)
    # cv.imshow('fusion', img_fusion)
    # img_fusion = cv.resize(img_fusion, (0,0), fx=2, fy=2)
    # print(img_fusion.shape)
    cv.imshow('fusion', img_fusion)
    return img_fusion


def make_image_fusion2(train_img_left, query_img_right):  # 没有仿射变换校准
    """ 定位两个图片的重复区域 """
    high, width = train_img_left.shape[:2]
    query_img, train_img = query_img_right, train_img_left
    query_pts, train_pts = fn_match_points(query_img.copy(), train_img.copy(), is_automatch=True, ishow=True)

    # src_pts = np.float32(query_pts)
    # dst_pts = np.float32(train_pts)
    # M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    # print(M)
    # train_img=result = cv.warpPerspective(train_img, M, (query_img.shape[1]+train_img.shape[1] , query_img.shape[0]))
    # cv.imshow('aaa', result)
    #
    # # 计算变换后的四个顶点坐标位置
    # train_pts = cv.perspectiveTransform(dst_pts, M)

    query_pts = np.array(query_pts, dtype=np.int32)
    train_pts = np.array(train_pts, dtype=np.int32)
    query_mean_pt = query_pts.mean(axis=0).astype(np.int32)
    train_mean_pt = train_pts.mean(axis=0).astype(np.int32)
    # print('匹配点为：', query_mean_pt, train_mean_pt)
    qx, qy, tx, ty = query_mean_pt[0], query_mean_pt[1], train_mean_pt[0], train_mean_pt[1]

    imga, imgb = query_img[:, qx:], train_img[:, :tx]
    # print('need fusion image shope=', imga.shape, imgb.shape)
    img_fusion = np.concatenate([imgb, imga], axis=1)

    if ty - qy > 0:
        # print('ty>qy')
        img_fusion[:-(ty - qy), :tx] = img_fusion[(ty - qy):, :tx]
        img_fusion[-(ty - qy):, :] = 0
        img_fusion = img_fusion[:-(ty - qy), :]
    elif ty - qy < 0:
        # print('ty<qy')
        img_fusion[-(ty - qy):, :tx] = img_fusion[:(ty - qy), :tx]
        img_fusion[:-(ty - qy), :] = 0
        img_fusion = img_fusion[-(ty - qy):, :]

    img_fusion = cv.resize(img_fusion, (width, high))

    # print(img_fusion.shape)
    # cv.imshow('fusion', img_fusion)
    # img_fusion = cv.resize(img_fusion, (0,0), fx=2, fy=2)
    # print(img_fusion.shape)
    cv.imshow('fusion', img_fusion)
    return img_fusion