import img_process as improc
import img_view as imview

# 图片预处理
apple_imgs = improc.read_imgs_from('../data/apples')
gray_apple_imgs = improc.imgs_graying(apple_imgs)
equ_apple_imgs = improc.imgs_hist_equalize(gray_apple_imgs)
imview.show_imgs(equ_apple_imgs[0:24])
edge_apple_imgs = improc.imgs_edge_detect(equ_apple_imgs, 30, 90)
imview.show_imgs(edge_apple_imgs[0:24])

# 获取苹果的特征参数


# 苹果病变判断，对科能存在缺陷的苹果使用训练模型得到病变类型(普通图像识别科能难以完成分类)


# 综合判断，得到苹果的分级结果
