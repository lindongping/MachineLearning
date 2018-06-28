import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

reviews = pd.read_csv('fandango_score_comparison.csv')
cols = ['FILM', 'RT_user_norm_round', 'Metacritic_user_norm_round', 'IMDB_norm_round']
norm_reviews = reviews[cols]
fig, ax = plt.subplots()
#bins指定柱形图一共有多少个柱子,range=(4,5)指定横轴的数据只要4-5之间的
ax.hist(norm_reviews['Metacritic_user_norm_round'],range=(4,5),bins=5)
plt.show()