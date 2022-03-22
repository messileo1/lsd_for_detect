import cv2
import math
import numpy as np

img = cv2.imread("images/02.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img',img)
lsd = cv2.createLineSegmentDetector(0, _scale=1)

dlines = lsd.detect(gray)
# print(dlines)
k = []
l = []
theta = []
ltheta = np.zeros((8))
suml = np.zeros((8))
meanl=np.zeros((8))
image = cv2.imread('images/02.jpg')
for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))
    # cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
    # cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
    # cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)
    k0 = (dline[0][3]-dline[0][1])/(dline[0][2]-dline[0][0])
    l0 = ((dline[0][2]-dline[0][0])**2+(dline[0][3]-dline[0][1])**2)**0.5
    theta0 = math.degrees(math.atan(k0))
    # theta0 = int(theta0*0.1)*10
    if theta0 >= 0:
        theta0 = (int(theta0 // 22.5)) * 22.5
    else:
        theta0 = (int(theta0 // 22.5)) * 22.5
    for i in range(0,8):
        if  theta0 == (i-4)*22.5:
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
print(k)

print(theta)
print(l)
print(meanl)
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
    k0 = (dline[0][3] - dline[0][1]) / (dline[0][2] - dline[0][0])
    theta0 = math.degrees(math.atan(k0))
    # theta0 = int(theta0 * 0.1) * 10

    if theta0 >= ( ((np.argmax(meanl))-4)*22.5 - 5 )and theta0 <= (((np.argmax(meanl))-3)*22.5 + 5) and l0 > 100:
    # if theta0 >= -30 and theta0 <= 0:
        cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)

        cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
        cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)
        sum = sum+1

        b = (x0*y1-x1*y0)/(x0-x1) #直线的截距
        jieju.append(b)
        x0list.append(x0)
        x1list.append(x1)
        y0list.append(y0)
        y1list.append(y1)

    else:
        pass
# cv2.line(image, (0, 200), (100, 100*int(math.tan((math.radians((((np.argmax(meanl))-4)*22.5 -10))))+200)), 255, 2, cv2.LINE_AA)
# cv2.line(image, (0, 200), (100, 100*int(math.tan((math.radians((((np.argmax(meanl))-3)*22.5 +10))))+200)), 255, 2, cv2.LINE_AA)
print('sum:',sum)
print('jieju:',jieju)












cv2.imshow('new',image)
cv2.waitKey()
# cv2.imwrite('LSD/line_dot10.jpg', image)
