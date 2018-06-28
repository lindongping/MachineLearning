import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree #构造决策树
import pydotplus
from PIL import Image
from sklearn.model_selection import train_test_split  #训练集和测试集区分
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV #用来选择合适的参数
from sklearn.ensemble import RandomForestRegressor #随机森林

#获取数据
housing = fetch_california_housing()
#print(housing.DESCR)
size = housing.data.shape    #(20640, 8)
#print(housing.data[0,:])

#构造决策树
dtr = tree.DecisionTreeRegressor(max_depth = 6)
dtr.fit(housing.data[:, [6, 7]], housing.target)#housing.target为分类标签

#要可视化显示 首先需要安装 graphviz   http://www.graphviz.org/Download..php
dot_data = \
    tree.export_graphviz(
        dtr,
        out_file = None,
        feature_names = housing.feature_names[6:8],
        filled = True,
        impurity = False,
        rounded = True
    )
#pip install pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
png = graph.create_png()
graph.write_png("decision_tree.png")
im = Image.open('decision_tree.png')
#im.show()

# 训练集和测试集区分，构造决策树并预测精度
data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
dtr = tree.DecisionTreeRegressor(random_state = 42)
dtr.fit(data_train, target_train)
accuracy_dtr = dtr.score(data_test, target_test)
print('决策树的精度为：', accuracy_dtr)

#随机森林
rfr = RandomForestRegressor( random_state = 42)
rfr.fit(data_train, target_train)
accuracy_rfr = rfr.score(data_test, target_test)
print('随机森林的精度为：', accuracy_rfr)

#选择合适的参数
tree_param_grid = { 'min_samples_split': list((3,6)),'n_estimators':list((50,100))}
#'min_samples_split': list((3,6,9))：以min_samples_split为标准，3,6，9哪个参数效果好？cv=5交叉验证5次
grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
cv_results_ = grid.cv_results_
best_params_ = grid.best_params_
best_score_ = grid.best_score_
print(cv_results_)
print(best_params_)
print(best_score_)

#按照选择的参数重新实验
rfr = RandomForestRegressor( min_samples_split=3,n_estimators = 100,random_state = 42)
rfr.fit(data_train, target_train)
accuracy_rfr2 = rfr.score(data_test, target_test)
print('选取参数后随机森林的精度为：', accuracy_rfr2)