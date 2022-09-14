import cv2
import numpy as np
import img_process as improc
import matplotlib.pyplot as plt

# 设置红色边界 (保留红色)
# H,S,V：色调，饱和度，明度
lower_1 = np.array([0, 43, 46])
upper_1 = np.array([10, 255, 255])
lower_2 = np.array([156, 43, 46])
upper_2 = np.array([180, 255, 255])

# 计算苹果的圆率
def clac_round_rate(imgs):
    gray_imgs = improc.imgs_graying(imgs)
    binary_imgs = improc.img_binarization(gray_imgs)
    contour_imgs = []
    roundnesses = []
    count = len(binary_imgs)
    for i in range(count):
        contours, hierarchy = cv2.findContours(binary_imgs[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcontour = contours[0]
        for contour in contours:
            if cv2.contourArea(maxcontour) < cv2.contourArea(contour):
                maxcontour = contour
        # 轮廓区域由函数cv.contourArea()给出。
        area = cv2.contourArea(maxcontour)
        # 弧长。可以使用cv.arcLength()函数找到它。第二个参数指定shape是闭合轮廓（如果传递True），还是只是曲线。
        arclen = cv2.arcLength(maxcontour, True)
        if arclen == 0:
            roundness = 0
        else:
            roundness = (4 * np.pi * area) / (arclen * arclen)
        roundnesses.append(roundness)
        # 要绘制轮廓，使用cv.drawContours函数。只要您有边界点，它也可以用于绘制任何形状。它的第一个参数是源图像，第二个参数是应该作为
        # 列表传递的轮廓，第三个参数是轮廓的索引（在绘制单个轮廓时很有用。要绘制所有轮廓，传递 - 1），其余参数是颜色，厚度等等
        contour_img = cv2.drawContours(imgs[i], [maxcontour], 0, (0, 255, 0), 2)
        contour_imgs.append(contour_img)
    return contour_imgs, roundnesses


# 封装图片显示函数
def image_show(image):
    plt.imshow(image)
    plt.show()


# 计算苹果的红色着色率
def clac_red_rate(imgs):
    red_rates = []
    red_imgs = []
    gray_imgs = improc.imgs_graying(imgs)
    binary_imgs = improc.img_binarization(gray_imgs)
    count = len(binary_imgs)
    for i in range(count):
        contours, __ = cv2.findContours(binary_imgs[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcontour = contours[0]
        for contour in contours:
            if cv2.contourArea(maxcontour) < cv2.contourArea(contour):
                maxcontour = contour
        # 轮廓区域由函数cv.contourArea()给出。
        area = cv2.contourArea(maxcontour)
        # 转换到HSV颜色空间
        img = cv2.copyTo(imgs[i], img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 获取掩码范围
        mask_1 = cv2.inRange(img_hsv, lower_1, upper_1)
        mask_2 = cv2.inRange(img_hsv, lower_2, upper_2)
        # 合并矩阵
        mask = cv2.bitwise_or(mask_1, mask_2)
        # 维度扩充并归一化
        mask = cv2.merge([mask, mask, mask])
        # 图像掩码处理
        img_red = img_hsv * mask
        # 转换为BGR通道
        img_red = cv2.cvtColor(img_red, cv2.COLOR_HSV2RGB)
        # 显示结果
        red_imgs.append(img_red)

    return red_imgs

# img_hsv = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)
# cv2.imshow("img_hsv", img_hsv)
# cv2.waitKey()
# img_hsv = img_hsv[:, :, 1]
# cv2.imshow("img_hsv", img_hsv)
# img_hsv[img_hsv < 50] = 0
# img_hsv[img_hsv >= 50] = 1
# cv2.imshow("img_hsv", img_hsv)
# if area == 0:
#     red_rates[i] = 0
# else:
#     red_rates[i] = np.sum(img_hsv) / area
#     if red_rates[i] > 1:
#         red_rates[i] = 1
