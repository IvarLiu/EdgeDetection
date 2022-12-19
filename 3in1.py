import cv2
import numpy as np
import matplotlib.pyplot as plt

def imageProcess(fullpath, name):
    img = cv2.imread(fullpath)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gaussianBlur = cv2.GaussianBlur(grayImage, (3,3), 0)
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    
    # Prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
    # Sobel
    x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Canny
    gaussian = cv2.GaussianBlur(grayImage, (5, 5), 0)
    Canny = cv2.Canny(gaussian, 50, 150)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    # SHOW
    plt.figure(dpi=300, figsize=(12.8,7.2))
    # plt.imshow(img_RGB),plt.title(name), plt.axis('off')
    plt.subplot(151),plt.imshow(img_RGB),plt.title(name), plt.axis('off')
    plt.subplot(152),plt.imshow(binary, cmap=plt.cm.gray ),plt.title('Binary'), plt.axis('off')
    plt.subplot(153),plt.imshow(Prewitt, cmap=plt.cm.gray ),plt.title('Prewitt'), plt.axis('off')
    plt.subplot(154),plt.imshow(Sobel, cmap=plt.cm.gray ),plt.title('Sobel'), plt.axis('off')
    plt.subplot(155),plt.imshow(Canny, cmap=plt.cm.gray ),plt.title('Canny'), plt.axis('off')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.show()

imageList = ['test2']
for i in imageList:
    path = r'C:\Users\36521\OneDrive\Desktop'
    filename = '\\' + i
    ext = '.jpg'
    input = path + filename + ext
    # output = path + filename + '_Edge' + ext
    imageProcess(input, i + ext)