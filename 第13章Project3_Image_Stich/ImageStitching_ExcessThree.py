from Stitcher import Stitcher
import cv2
import my_utils
# 只拼接两张图片

# 读取需要拼接的图片
# imageA_original = cv2.imread("left_01.png")
# imageB_original = cv2.imread("right_01.png")
imageA_original = cv2.imread("left_01.jpg")
imageB_original = cv2.imread("right_01.jpg")
imageC_original = cv2.imread("right_02.jpg")

# 图像预处理-改变图像大小
imageA = my_utils.resize(imageA_original,width=500)
imageB = my_utils.resize(imageB_original,width=500)
imageC = my_utils.resize(imageC_original,width=500)

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
(result, vis) = stitcher.stitch([result, imageC], showMatches=True)

# 显示所有图片
stitcher.cv_show("Image A", imageA)
stitcher.cv_show("Image B", imageB)
stitcher.cv_show("Image C", imageC)
stitcher.cv_show("Keypoint Matches", vis)
stitcher.cv_show("Result", result)