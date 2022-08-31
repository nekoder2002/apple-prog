import cv2
import numpy as np
import img_process as improc


# 计算苹果的红色着色率
def clac_red_rate():
    pass

#更新

# 计算苹果的圆率
def clac_round_rate(imgs):
    gray_imgs = improc.imgs_graying(imgs)
    binary_imgs = improc.img_binarization(gray_imgs)
    contour_imgs = []
    roundnesses = []
    count = len(binary_imgs)
    for i in range(count):
        contours, hierarchy = cv2.findContours(binary_imgs[i], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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