import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


f1 = 0.027
f2 = 0.054

y = np.linspace(0, 0.5, 200)
x = (f1*(f1-f2))/(y*f2-f1)



plt.plot(x, y)

plt.title('r1/r2与深度Z的关系')
plt.xlabel('Z')
plt.ylabel('r1/r2')

plt.show()