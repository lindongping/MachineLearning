import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

reviews = pd.read_csv('fandango_score_comparison.csv')
#print(reviews.head())
cols = ['FILM', 'RT_user_norm_round', 'Metacritic_user_norm_round', 'IMDB_norm_round']
norm_reviews = reviews[cols]
#print(norm_reviews[:1])
num_cols = ['RT_user_norm_round', 'Metacritic_user_norm_round', 'IMDB_norm_round']

bar_heights = norm_reviews.ix[0, num_cols].values#条形图中数据柱的高度
bar_position = np.arange(3) + 0.75#每个柱离0点的距离，柱之间的距离最好相等
tick_position = range(0,3)

fig, ax = plt.subplots()#ax用来画图，fig进行一些操作，如图的样式
ax.bar(bar_position, bar_heights, 0.3)#画条形图,0.3为柱宽度
#ax.bar(bar_position, bar_heights, 0.3) #横着的条形图
ax.set_xticks(tick_position)
ax.set_xticklabels(num_cols, rotation = 45)
ax.set_xlabel('dfdf')
ax.set_ylabel("dsdf")
ax.set_title('dsdsf')
#ax.set_xlim(0,50)
plt.show()

