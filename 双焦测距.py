import cv2
import math
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

def ceju(input,output):



    img = cv2.imread(input, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(1000,700))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img)
    lsd = cv2.createLineSegmentDetector(0, _scale=1)

    dlines = lsd.detect(gray)
    k = []
    l = []
    theta = []
    ltheta = np.zeros((8))
    suml = np.zeros((8))
    meanl=np.zeros((8))
    image = img
    image2 =img
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        # cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        # cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
        # cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)
        k0 = (dline[0][3] - dline[0][1]) / (dline[0][2] - dline[0][0])
        l0 = ((dline[0][2] - dline[0][0]) ** 2 + (dline[0][3] - dline[0][1]) ** 2) ** 0.5
        theta0 = math.degrees(math.atan(k0))
        # theta0 = int(theta0*0.1)*10
        if theta0 >= 0:
            theta0 = (int(theta0 // 22.5)) * 22.5
        else:
            theta0 = (int(theta0 // 22.5)) * 22.5
        for i in range(0, 8):
            if theta0 == (i - 4) * 22.5:
                ltheta[i] += l0
                suml[i] += 1

                break
            else:
                pass

        k.append(k0)
        l.append(l0)
        theta.append(theta0)
    meanl = ltheta / suml
    where_are_NaNs = np.isnan(meanl)
    meanl[where_are_NaNs] = 0
    # print('k',k)
    #
    # print('theta',theta)
    # print('l',l)
    # print('meanl',meanl)
    print(np.argmax(meanl))
    sum = 0
    jieju = []
    x0list = []
    x1list = []
    y0list = []
    y1list = []

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        l0 = ((dline[0][2] - dline[0][0]) ** 2 + (dline[0][3] - dline[0][1]) ** 2) ** 0.5
        theta0 = math.degrees(math.atan((dline[0][3] - dline[0][1]) / (dline[0][2] - dline[0][0])))
        # theta0 = int(theta0 * 0.1) * 10

        if theta0 >= (((np.argmax(meanl)) - 4) * 22.5 - 10) and theta0 <= (((np.argmax(meanl)) - 3) * 22.5 + 10) and l0 > 105:
            # if theta0 >= -30 and theta0 <= 0:
            # cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
            # cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
            # cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)
            sum = sum + 1

            b = int((x0 * y1 - x1 * y0) / (x0 - x1) ) # 直线的截距
            jieju.append(b)
            x0list.append(x0)
            x1list.append(x1)
            y0list.append(y0)
            y1list.append(y1)

        else:
            pass
    # print('sum:', sum)
    # print('jieju:', jieju)


    X = [[i] for i in jieju]
    #method是指计算类间距离的方法,比较常用的有3种:
    #single:最近邻,把类与类间距离最近的作为类间距
    #average:平均距离,类与类间所有pairs距离的平均
    #complete:最远邻,把类与类间距离最远的作为类间距
    Z = linkage(X, 'single')
    f = fcluster(Z,30,'distance')
    # print(f)
    fig = plt.figure(figsize=(5, 3))
    dn = dendrogram(Z)
    plt.show()


    yc = 350
    xc = 500
    dis =[]

    for i in range (1,f.max()+1):
        xzanshi = []
        yzanshi = []

        for j in range(len(jieju)):
            if f[j] == i:
                xzanshi.append(x0list[j])
                yzanshi.append(y0list[j])
                xzanshi.append(x1list[j])
                yzanshi.append(y1list[j])
        print(xzanshi)
        print(yzanshi)
        k = np.polyfit(xzanshi, yzanshi, 1)
        # print('k=',k)
        cv2.line(image2, (0, int(k[1])), (3500, int(3500*k[0]+k[1])), (0,0,255), 10, cv2.LINE_AA)
        distance = (abs(k[0]*xc-yc+k[1]))/(math.sqrt(k[0]*k[0]+1))
        dis.append(distance)

    print('distance',dis)
    val = min(dis)
    # print('val:',val)
    cv2.imshow('new', image2)
    # cv2.imwrite('result/11.jpg',image2)
    cv2.waitKey()
    cv2.imwrite(output,image2)
    return val

# cv2.imwrite('LSD/line_dot10.jpg', image)



if __name__ == "__main__":
    # r1 = ceju('ceju/01.jpg','01.jpg')
    # print("r1",r1)
    # r2 = ceju('ceju/02.jpg','02.jpg')
    # print("r2", r2)
    f1 =27
    f2 =54
    r1 = 245
    r2 = 500
    print(r1/r2)
    depth = (r2*f1*(f1-f2))/(r1*f2-r2*f1)
    print('depth:',depth)