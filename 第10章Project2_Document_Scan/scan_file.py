import numpy as np
import argparse
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
print(args)


# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取图片数据
image = cv2.imread(args['image'])
# cv_show('image',image)
print(image.shape)
# 将要对图像进行大小变化，先保存一下变化率
ratio = image.shape[0] / 500.0
# 获得原始图像
orig = image.copy()


# 按比例变化图像大小
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    # 当宽度没有进行指定时，按照高度的变化进行比例变化
    # 当高度没有进行指定时，按照宽度的变化进行比例变化
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        print('width is None', dim)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
        print('height is None', dim)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


image = resize(orig, height=500)
print(image.shape)
# cv_show('image',image)

# 对图像进行预处理操作
# 转灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)
# 去噪操作
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# cv_show('gray',gray)

# 进行Canny边缘检测
edged = cv2.Canny(gray, 75, 200)
print(edged.shape)
# cv_show('Canny_edged',edged)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv_show('image', image)
cv_show('Canny_edged', edged)

# 再进行轮廓检测
# RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中
# CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
binary, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
draw_img = image.copy()
res = cv2.drawContours(draw_img, cnts, -1, (0, 0, 255), 2)
cv_show('find_Cnts', draw_img)

# 根据轮廓面积排序选择最大的五个轮廓
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
draw_img = image.copy()
res = cv2.drawContours(draw_img, cnts, -1, (0, 0, 255), 2)
cv_show('find_Cnts', res)

# 遍历这五个最大的轮廓
for c in cnts:
    # 计算轮廓近似
    # 由于有些轮廓比较粗糙或者太详细，可以改变近似算法的阈值来调整
    peri = cv2.arcLength(c, True)
    # c:输入的点集
    # epsilon：从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    # True ： 表示封闭
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 有4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break
# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv_show('Get_Cnts', image)


# 透视变换
# 给出四组坐标值，坐标不共线
def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)


    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    # 只能是矩形形状较好

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


warped =four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

cv_show("warped",warped)

# 二值处理
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
# cv_show('Binary_warped',warped)

# 保存图像
cv2.imwrite('scan.jpg',ref)

# 展示结果
print("STEP 3: 变换")
cv_show("Original", resize(orig, height = 650))
cv_show("Scanned", resize(ref, height = 650))