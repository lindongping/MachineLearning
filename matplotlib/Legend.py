import matplotlib.pyplot as plt
import pandas as pd

unrate = pd.read_csv('unrate.csv')
unrate['DATE'] = pd.to_datetime(unrate['DATE']) #转换成标准日期格式
unrate['MONTH'] = unrate['DATE'].dt.month #转成月份,构造month列
#print(unrate.head())

colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(5):
    start_index = 12 * i
    end_index = 12 * (i+1)
    subset = unrate[start_index:end_index]
    label = str(1948 + i)
    plt.plot(subset['MONTH'], subset['VALUE'], c = colors[i], label = label)#label = label
plt.legend(loc = 'best')#调用方法
plt.show()