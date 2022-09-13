import matplotlib.pyplot as plt
import cv2


# 以cv窗口展示单个图片
def show_img_win(win_name, img):
    cv2.namedWindow(win_name, 0)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 展示一组图片
def show_imgs(imgs):
    total_num = len(imgs)
    row_num = (total_num + 5) // 6
    plt.figure(figsize=(36, row_num * 6))
    for i in range(total_num):
        plt.subplot(row_num, 6, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()