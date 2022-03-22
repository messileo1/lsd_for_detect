import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('biaoding2/*.jpg')
i = 0
for fname in images:
    img = cv2.imread(fname)
    # img = cv2.resize(img,(1600,1200))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,8),None)
    print(ret)
    print(img.shape)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6,8), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(50)
        # cv2.imwrite("biaoding_result/%d.jpg"%(i+1),img)
        i += 1
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("ret:",ret)
print("内参:",mtx)
print("畸变系数:",dist)
print("旋转矩阵:",rvecs)
print("平移矩阵:",tvecs)




img = cv2.imread('biaoding2/10.jpg')
h,w = img.shape[:2]
# print(h,w)
newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print('新内参矩阵：',newcameramtx)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# cv2.imshow('dst',dst)
# cv2.waitKey()
# crop the image
x,y,w,h = roi
# print(roi)
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.jpg',dst)