import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\36521\OneDrive\Desktop\A1_EdgeDetection\bmps'
filename = r'\Barbara'
ext = '.bmp'
input = path + filename + ext
output = path + filename + '_cannyEdge' + ext 

# 读取图像
img = cv2.imread(input)
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# 高斯滤波降噪
gaussian = cv2.GaussianBlur(grayImage, (5, 5), 0)
 
# Canny算子
Canny = cv2.Canny(gaussian, 50, 150)
 
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
 
# 显示图形
# titles = [u'原始图像', u'Canny算子']
# images = [img, Canny]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

plt.subplot(121),plt.imshow(img_RGB),plt.title('Original'), plt.axis('off') #坐标轴关闭
plt.subplot(122),plt.imshow(Canny, cmap=plt.cm.gray ),plt.title('Canny'), plt.axis('off')
plt.show()
cv2.imwrite(output, Canny)