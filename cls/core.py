import img_process as improc
import img_view as imview
import extract_feature as ef
import matplotlib.pyplot as plt


def image_show(image):
    plt.imshow(image)
    plt.show()


# 图片预处理
apple_imgs = improc.read_imgs_from('../data/apples')
# gray_apple_imgs = improc.imgs_graying(apple_imgs)
# equ_apple_imgs = improc.imgs_hist_equalize(gray_apple_imgs)
# denoise_apple_imgs = improc.imgs_denoise(equ_apple_imgs, 'GaussianBlur', (3, 3))
# edge_apple_imgs = improc.imgs_edge_detect(denoise_apple_imgs, 30, 90)
# binary_apple_imgs = improc.img_binarization(gray_apple_imgs)
# imview.show_imgs(binary_apple_imgs[0:24])
# # 获取苹果的特征参数
#
# # 圆度
# apple_contours, apple_roundness_list = ef.clac_round_rate(apple_imgs)
# imview.show_imgs(apple_contours[0:24])
# 红色着色率
red_imgs = ef.clac_red_rate(apple_imgs)
imview.show_imgs(red_imgs[0:24])

# 苹果病变判断，对科能存在缺陷的苹果使用训练模型得到病变类型(普通图像识别科能难以完成分类,神经网诺判断)
# 综合判断，得到苹果的分级结果
