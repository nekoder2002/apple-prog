import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 封装图片显示函数
def image_show(image):
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # 读取原图
    img_lenna = cv.imread('../data/apples/img1_1.jpg')
    cv.imshow("img_lenna", img_lenna)

    # 转换到HSV颜色空间
    img_hsv = cv.cvtColor(img_lenna, cv.COLOR_BGR2HSV)
    image = cv.cvtColor(img_hsv, cv.COLOR_BGR2RGB)
    cv.imshow("image", image)

    # 设置红色边界 (保留红色)
    # H,S,V：色调，饱和度，明度
    lower_1 = np.array([0, 43, 0])
    upper_1 = np.array([10, 255, 255])

    lower_2 = np.array([156, 43, 0])
    upper_2 = np.array([180, 255, 255])

    # 获取掩码范围
    mask_1 = cv.inRange(img_hsv, lower_1, upper_1)
    mask_2 = cv.inRange(img_hsv, lower_2, upper_2)

    # 合并矩阵
    mask = cv.bitwise_or(mask_1, mask_2)

    # 维度扩充并归一化
    mask = cv.merge([mask, mask, mask]) // 255

    # 图像掩码处理
    img_red = img_hsv * mask

    # 转换为BGR通道
    img_bgr = cv.cvtColor(img_red, cv.COLOR_HSV2BGR)

    # 显示结果

    image_show(img_bgr)
