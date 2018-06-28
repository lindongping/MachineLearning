import pandas as pd
import matplotlib.pyplot as plt

reviews = pd.read_csv('fandango_score_comparison.csv')
cols = ['FILM', 'RT_user_norm_round', 'Metacritic_user_norm_round', 'IMDB_norm_round']
norm_reviews = reviews[cols]

fig, ax = plt.subplots()
ax.scatter(norm_reviews['RT_user_norm_round'], norm_reviews['Metacritic_user_norm_round'])
ax.set_xlabel('dfdf')
ax.set_ylabel("dsdf")
ax.set_title('dsdsf')
plt.show()