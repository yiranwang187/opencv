import numpy as np
# 用来处理命令行参数的库
import argparse
import cv2
import myutils
from imutils import contours

# 设置参数
ap = argparse.ArgumentParser()
# print(ap)
# 添加命令行命令
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-t', '--template', required=True, help='path to template OCR-A image')
# vars()返回对象object的属性和属性值的字典对象
args = vars(ap.parse_args())
print(args)
# print(help(vars))
# print(help(ap.add_argument))

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# cv绘图显示
def cv_show(name, img):
    cv2.imshow(name, img)
    # 等待键盘输入，当键盘输入函数时，程序继续运行
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取模板图像
img = cv2.imread(args['template'])
print(img)
# cv_show('img', img)

# 转换为灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('ref', ref)

# 转换为二值图像
# cv2.THRESH_BINARY_INV:
# 超过阈值(10)部分取0，否则取maxval（最大值）
_, ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)
print(ref)
# cv_show('ref', ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,
# cv2.RETR_EXTERNAL只检测外轮廓，
# cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
# 获得模板的轮廓
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ref_, refCnts_none, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print(refCnts)
print(refCnts_none)

# 绘制所有轮廓
img_simple = cv2.drawContours(img.copy(), refCnts, -1, (0, 0, 255), 2)
img_none = cv2.drawContours(img.copy(), refCnts_none, -1, (0, 0, 255), 2)
# 显示绘制的轮廓
vres = np.vstack((img, img_simple, img_none))
cv_show('img',vres)

print(type(refCnts))
print(np.array(refCnts, dtype=object).shape)


# 为什么要进行排序？
# 当找到十个轮廓时，轮廓的存储方式并不是按照先后排序的，也就是说轮廓没有默认按照从左到右的方式进行排序
# 为了能够得到有顺序的轮廓所以需要进行排序
# 排序后的轮廓才可以进行相应的赋值操作
# 排序，从左到右，从上到下
refCnts_sort = myutils.sort_contours(refCnts, method="left-to-right")[0]
img_unsort = cv2.drawContours(img.copy(), refCnts[0:3], -1, (0, 0, 255), 2)
img_sort = cv2.drawContours(img.copy(), refCnts_sort[0:3], -1, (0, 0, 255), 2)
vres = np.vstack((img, img_unsort, img_sort))
cv_show('img',vres)



digits = {}
# 遍历每一个轮廓
for (i, c) in enumerate(refCnts_sort):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # 每一个数字对应每一个模板
    digits[i] = roi
cv_show('img_digit',digits[0])

# 初始化卷积核
# 构造一个长方形的核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# print(rectKernel)
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# print(sqKernel)

# 读取输入图像，进行预处理
image = cv2.imread(args['image'])
# print(args['image'])
# print(image)
# cv_show('process_image',image)
print(type(image))
print(image.shape)

# 改变图片大小
image = myutils.resize(image, width=300)
# cv_show('process_image',image)
print(type(image))
print(image.shape)

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('gray',gray)

# 礼帽操作：原始输入-(先腐蚀，再膨胀)
# 突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 计算图像梯度，在这使用Sobel算子x方向效果更佳
# ksize=-1相当于用3*3的
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# 归一化
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
# 为提取数字方框做准备
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)

# THRESH_OTSU会自动寻找合适的阈值，适合双峰
# 需把阈值参数设置为0
# 将图像转换为二值图像
thresh = cv2.threshold(gradX, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 再来闭操作（先膨胀，再腐蚀）
# 填充数字框里面的空隙
thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)  # 再来一个闭操作
vres = np.vstack((thresh, thresh_close))
cv_show('One_Morph_Close_and_Two_Morph_Close', vres)

# 计算轮廓,找到数字的轮廓
thresh_, threshCnts, hierarchy = cv2.findContours(thresh_close.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
# 在原图像上绘制轮廓，因为在对进行过形态学上的图片只有二值，不好绘制
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)

# 找到数字所在的矩形位置
# 根据轮廓特征找到我们所需要的轮廓
locs = []
for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择合适的区域，
    # 根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))

# 再从左到右进行排序

locs = sorted(locs, key=lambda x: x[0])

# 再遍历轮廓中的每个数字
output = []
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 初始化一组数字
    groupOutput = []

    # 根据坐标提取每个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    # 预处理-变为二值图
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一组的轮廓
    group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 对轮廓进行排序，从左到右
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
    # print(type(digitCnts))
    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        # 字典的遍历(key ,value)
        # 返回对应的数字和数字图像
        for (digit, digitROI) in digits.items():
            print(digit, digitROI.shape)
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
        print('groupOutput:', groupOutput)

    # 画出来-画一组数字的框
    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    # 画一组图像数字对应的数字
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
