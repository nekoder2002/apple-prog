# Apple_Prog



## 项目结构

    |apple-prog
	    |cls 苹果分级分类模块
            img_process.py  图像处理
            img_view.py 图像可视化
            extract_feature.py 特征提取
            core.py 核心文件
	    |web web模块
	    |build 训练模型模块
	    |models 存放训练好的预测模型
	    |data 存放苹果数据
	        | apples 苹果数据（测试用,，以后可能会修改）



## 详细内容

### cls 苹果分级分类模块

#### img_process.py

封装了图像处理相关函数

```
# 读取路径下的所有图片
read_imgs_from(dir_path)

# 图片灰度化处理
imgs_graying(imgs)

# 图片直方图均衡化
imgs_hist_equalize(imgs)

# 图片去噪 (MeanBlur,MediumBlur,GaussianBlur)
imgs_denoise(imgs, method_name, ksize)

# 图片边缘检测 (Canny算法已进行Gaussian平滑去噪)
def imgs_edge_detect(imgs, low_threshold, high_threshold)
```



#### img_view.py

封装了图像可视化相关函数

```
# 以cv窗口展示单个图片
show_img_win(win_name, img)

# 展示一组图片
show_imgs(imgs)
```



#### extract_feature.py

特征提取相关函数

```
# 计算苹果的红色着色率
clac_red_rate()

# 计算苹果的圆率
clac_round_rate(imgs)
```