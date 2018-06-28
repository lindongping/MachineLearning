import pandas as pd
import matplotlib.pyplot as plt

unrate = pd.read_csv('unrate.csv')
unrate['DATE'] = pd.to_datetime(unrate['DATE']) #转换成标准日期格式
#print(unrate.head())
data_15 = unrate[0:15]
#print(data_15)
plt.plot(data_15['DATE'], data_15['VALUE'])
plt.xticks(rotation=45)#指定x轴坐标倾斜45度
plt.xlabel('Month')
plt.ylabel('Unemployment_rate')
plt.title('Unemployment_rate diagram')
plt.show()