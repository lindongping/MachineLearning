import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(6,6))#figsize=(6,6)指定画图区域大小，默认为fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,4)

ax1.plot(np.random.randint(1,5,5),np.arange(5))
plt.show()