import img_process as improc
import img_view as imview

# 图片预处理
apple_imgs = improc.read_imgs_from('data/apples')
gray_apple_imgs = improc.imgs_graying(apple_imgs)
equ_apple_imgs = improc.imgs_hist_equalize(gray_apple_imgs)
imview.show_imgs(equ_apple_imgs[0:72])