import pandas
import numpy as np
import matplotlib.pyplot as plt

titanic = pandas.read_csv("titanic_train.csv")
#print(titanic.describe().to_string())  #to_string()完全显示
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())#处理缺失数据
#print(titanic['Sex'].unique())['male' 'female']
titanic.loc[titanic['Sex']=='male','Sex'] = 0#将非数字数据转换成数字
titanic.loc[titanic['Sex']=='female','Sex'] = 1
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked']=='S','Embarked'] = 0
titanic.loc[titanic['Embarked']=='C','Embarked'] = 1
titanic.loc[titanic['Embarked']=='Q','Embarked'] = 2

#线性回归算法
from sklearn.linear_model import LinearRegression
alg = LinearRegression()
#交叉验证
from sklearn.model_selection import KFold
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
#将训练/测试数据集划分n_splits个互斥子集，每次用其中一个子集当作验证集，
#剩下的n_splits-1个作为训练集，进行n_splits次训练和测试，得到n_splits个结果
#n_splits：表示划分几等份
#shuffle：在每次划分时，是否进行洗牌,为Falses时，其效果等同于random_state等于整数，每次划分的结果相同
kf = KFold(n_splits=3,random_state=1)

predictions = []  #交叉验证每一次的预测结果
ave_predictions = []   #交叉验证每一次预测结果的平均值
#kf.split(X, y=None, groups=None)：将数据集划分成训练集和测试集，返回索引生成器
for train, test in kf.split(titanic):
    train_prdictors = titanic[predictors].iloc[train,:]  #训练集数据
    train_target = titanic['Survived'].iloc[train]     #训练集标签
    alg.fit(train_prdictors, train_target)  #用线性回归进行训练
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])    #用测试集进行预测
    #test_predictions为测试集中每一个乘客的预测值，为一个列表
    predictions.append(test_predictions)
    #计算每次交叉验证的预测结果平均值
    sum = 0
    for i in test_predictions:
        sum = sum + i
    ave = sum / len(test_predictions)
    ave_predictions.append(ave)
#交叉验证的平均值
sum = 0
for i in ave_predictions:
    sum = sum + i
ave = sum / len(ave_predictions)
print('交叉验证结果的平均值为：%.4f'%ave)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
accuracy = len(predictions[predictions == titanic['Survived']]) / len(predictions)
print('交叉验证结果的准确率为（通过线性回归模型预测）：%.4f'%accuracy)

#逻辑回归算法
from sklearn.linear_model import LogisticRegression
alg2 = LogisticRegression(random_state=1)
predictions2 = []  #交叉验证每一次的预测结果
for train, test in kf.split(titanic):
    train_prdictors = titanic[predictors].iloc[train,:]  #训练集数据
    train_target = titanic['Survived'].iloc[train]     #训练集标签
    alg2.fit(train_prdictors, train_target)  #用线性回归进行训练
    test_predictions = alg2.predict(titanic[predictors].iloc[test,:])    #用测试集进行预测
    predictions2.append(test_predictions)
predictions2 = np.concatenate(predictions2, axis=0)
accuracy2 = len(predictions2[predictions2 == titanic['Survived']]) / len(predictions2)
print('交叉验证结果的准确率为（通过逻辑回归模型预测）：%.4f'%accuracy2)


from sklearn.ensemble import RandomForestClassifier
#构造随机森林模型
#n_estimators森林由多少树组成；min_samples_split样本最小切分点；min_samples_leaf叶子节点最少数量
alg3 = RandomForestClassifier(random_state=1, n_estimators=80, min_samples_split=4, min_samples_leaf=2)
predictions3 = []  #交叉验证每一次的预测结果
for train, test in kf.split(titanic):
    train_prdictors = titanic[predictors].iloc[train,:]  #训练集数据
    train_target = titanic['Survived'].iloc[train]     #训练集标签
    alg3.fit(train_prdictors, train_target)  #用线性回归进行训练
    test_predictions = alg3.predict(titanic[predictors].iloc[test,:])    #用测试集进行预测
    predictions3.append(test_predictions)
predictions3 = np.concatenate(predictions3, axis=0)
predictions3[predictions3 > 0.5] = 1
predictions3[predictions3 <= 0.5] = 0
accuracy3 = len(predictions3[predictions3 == titanic['Survived']]) / len(predictions3)
print('交叉验证结果的准确率为（通过随机森林模型预测）：%.4f'%accuracy3)

#添加额外两个指标
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

#特征重要性分析
from sklearn.feature_selection import SelectKBest, f_classif
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize","NameLength"]
# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
# Pick only the four best features.
#predictors2 = ["Pclass", "Sex", "Fare"]

