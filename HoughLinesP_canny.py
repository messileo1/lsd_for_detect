import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import math



# def gray_weightmean_rgb(wr,wg,wb,inputimagepath,windowname,outimagepath):
#     img = cv2.imread(inputimagepath)
#     gray_weightmean_rgb_image = img.copy()
#     img_shape = img.shape
#     for i in range(img_shape[0]):
#         for j in range(img_shape[1]):
#             gray_weightmean_rgb_image[i,j] = (int(wr*img[i,j][2])+int(wg*img[i,j][1])+int(wb*img[i,j][0]))/3
#     print(gray_weightmean_rgb_image)
#     cv2.namedWindow(windowname)  #控制显示图片窗口的名字
#     cv2.imshow(windowname, gray_weightmean_rgb_image)#显示灰度化后的图像
#     cv2.imwrite(outimagepath, gray_weightmean_rgb_image)  # 保存当前灰度值处理过后的文件
#     cv2.waitKey()#等待操作
#     cv2.destroyAllWindows()#关闭显示图像的窗口
#     return gray_weightmean_rgb_image



def img_processing(img):
    # 灰度化
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = gray_weightmean_rgb(0.6,0.15,0.15,input,'windowname','gray.jpg')
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # canny边缘检测
    edges = cv2.Canny(img, 100,150, apertureSize=3)
    cv2.imshow('ed',edges)
    cv2.waitKey()
    return edges


def line_detect(img):
    img = Image.open(img)
    # img = ImageEnhance.Contrast(img).enhance(3)#增强图像
    # img.show()
    img = np.array(img)
    result = img_processing(img)
    # 霍夫线检测
    lines = cv2.HoughLinesP(result, 1, 1* np.pi / 180, 60, minLineLength=10, maxLineGap=20)
    # print(lines)
    print("Line Num : ", len(lines))
    image  = cv2.imread('images/09.jpg')
    # 画出检测的线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(image,(x1, y1),2,(0,255,0),-1)
            cv2.circle(image, (x2, y2), 2, (0, 255, 0), -1)
        pass
    img = Image.fromarray(image, 'RGB')
    cv2.imwrite('Hough_cannyorigin/Hough_dot01.jpg',image)
    img.show()


if __name__ == "__main__":
    input = "images/09.jpg"
    line_detect(input)

    pass