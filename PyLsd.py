import cv2


img = cv2.imread("images/12.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lsd = cv2.createLineSegmentDetector(0, _scale=1)

dlines = lsd.detect(gray)
image = cv2.imread('images/12.jpg')
for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))
    cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
    # cv2.circle(image, (x0, y0), 2, (0, 255, 0), -1)
    # cv2.circle(image, (x1, y1), 2, (0, 255, 0), -1)


cv2.imshow('new',image)
cv2.waitKey()
# cv2.imwrite('LSD/line10.jpg', image)
