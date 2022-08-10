# Apple_Prog项目



## 结构

img_process.py  图像处理相关脚本

img_view.py 图像可视化相关脚本

main.py 主程序

data 存放数据

| apples 苹果数据（测试用）

## img_process.py

```python
# 读取路径下的所有图片
def read_imgs_from(dir_path)

# 图片灰度化处理
def imgs_graying(imgs)

# 图片直方图均衡化
def imgs_hist_equalize(imgs)

# 图片去噪 (MeanBlur,MediumBlur,GaussianBlur)
def imgs_denoise(imgs, method_name, ksize)
```



## img_view.py

```python
# 以cv窗口展示单个图片
def show_img_win(win_name, img)

# 展示一组图片
def show_imgs(imgs)
```
