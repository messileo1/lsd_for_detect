import os
import numpy as np
import cv2
from PIL import Image
import math




def img_processing(img):
    # 灰度化
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = gray_weightmean_rgb(0.6,0.15,0.15,input,'windowname','gray.jpg')
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # canny边缘检测
    edges = cv2.Canny(img, 150,150, apertureSize=3)
    cv2.imshow('ed',edges)
    cv2.waitKey()
    return edges





def line_detect(img):
    img = cv2.imread(input)
    # img = cv2.resize(img,(960,544))
    img1 = img
    # img = ImageEnhance.Contrast(img).enhance(3)#增强图像
    # img.show()
    img = np.array(img)
    result = img_processing(img)
    # 霍夫线检测
    lines = cv2.HoughLinesP(result, 1, 1* np.pi / 180, 60, minLineLength=10, maxLineGap=20)
    # print(lines)
    print("Line Num : ", len(lines))
    image  = cv2.imread("./images/08.jpg")
    k = []
    l = []
    theta = []
    ltheta = np.zeros((6))
    suml = np.zeros((6))
    meanl = np.zeros((6))
    # 画出检测的线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # cv2.circle(image,(x1, y1),2,(0,255,0),-1)
            # cv2.circle(image, (x2, y2), 2, (0, 255, 0), -1)
            k0 = (y2 - y1) / (x2 -x1)
            l0 = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
            theta0 = math.degrees(math.atan(k0))
            if theta0 >= 0:
                theta0 = (int(theta0 // 30)) * 30
            else:
                theta0 = (int(theta0 // 30) ) * 30
            for i in range(0, 6):
                if theta0 == (i - 3) * 30:
                    ltheta[i] += l0
                    suml[i] += 1

                    break
                else:
                    pass

            k.append(k0)
            l.append(l0)
            theta.append(theta0)
        pass
    meanl = ltheta / suml
    where_are_NaNs = np.isnan(meanl)
    meanl[where_are_NaNs] = 0
    img = Image.fromarray(image, 'RGB')
    # cv2.imwrite('Hough_cannyorigin/Hough_dot01.jpg',image)
    img.show()
    print(theta)
    print(meanl)
    print(np.argmax(meanl))


    for line in lines:
        for x1, y1, x2, y2 in line:
            k0 = (y2 - y1) / (x2 - x1)
            theta0 = math.degrees(math.atan(k0))
            l0 = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
            if theta0 >= ( ((np.argmax(meanl))-3)*30-10  )and theta0 <= (((np.argmax(meanl))-2)*30 +10 ) :
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.circle(image,(x1, y1),2,(0,255,0),-1)
                # cv2.circle(image, (x2, y2), 2, (0, 255, 0), -1)
    img = Image.fromarray(image, 'RGB')
    cv2.imwrite('Hough_result/Houghy08.jpg',image)
    img.show()






if __name__ == "__main__":
    input = "./sobel/erzhi08.jpg"

    # input = "./sobel/erzhi01.jpg"
    line_detect(input)

    pass