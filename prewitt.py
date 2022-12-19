import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\36521\OneDrive\Desktop\A1_EdgeDetection\bmps'
filename = r'\Barbara'
ext = '.bmp'
input = path + filename + ext
output = path + filename + '_prewittEdge' + ext 

# 读取图像
img = cv2.imread(input)
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
 
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
 
# 显示图形
plt.subplot(121),plt.imshow(img_RGB),plt.title('Original'), plt.axis('off') #坐标轴关闭
plt.subplot(122),plt.imshow(Prewitt, cmap=plt.cm.gray ),plt.title('Prewitt'), plt.axis('off')
plt.show()
cv2.imwrite(output, Prewitt)