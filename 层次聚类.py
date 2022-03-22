#
# from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
# from matplotlib import pyplot as plt
# X = [[i] for i in [71.00350262697023, 126.0, 68.0, 71.63546798029557, 128.0, 97.0, 100.0, 100.83870967741936, 105.69005847953217]]
# #method是指计算类间距离的方法,比较常用的有3种:
# #single:最近邻,把类与类间距离最近的作为类间距
# #average:平均距离,类与类间所有pairs距离的平均
# #complete:最远邻,把类与类间距离最远的作为类间距
# Z = linkage(X, 'single')
# f = fcluster(Z,10,'distance')
# print(f)
# fig = plt.figure(figsize=(5, 3))
# dn = dendrogram(Z)
#
# plt.show()

import cv2


# 开启ip摄像头
video = "rtsp://admin:admin111@10.192.4.167:554/Streaming/Channels/101"  # 此处@后的ipv4 地址需要改为app提供的地址
cap = cv2.VideoCapture(video)

while True:
    # Start Camera, while true, camera will run

    ret, image_np = cap.read()

    # Set height and width of webcam
    height = 600
    width = 1000

    # Set camera resolution and create a break function by pressing 'q'
    cv2.imshow('object detection', cv2.resize(image_np, (width, height)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
