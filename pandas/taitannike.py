import pandas as pd
import numpy as np

titanic_train = pd.read_csv('titanic_train.csv')
#print(titanic_train.head())
print('-------'*3)

#检测年龄为空的数据
age = titanic_train['Age']
age_isnull = pd.isnull(age)
#print(age_isnull)
age_null = age[age_isnull]
print(age_null)
print(len(age_null))
print('-------'*3)

#计算平均年龄，去掉缺失值
good_age = age[age_isnull == False]
ave_age = sum(good_age)/len(good_age)
print(ave_age)
# ave_age2 = age.mean()   #直接用自带函数也可以求均值
# print(ave_age2)
#直接去掉缺失值对应的行
#titanic_train.dropna(axis=1)
#titanic_train.dropna(axis=0,subset=['Age','Sex'])
print('-------'*3)

#按照类别进行相应的计算
#根据船舱等级Pclass计算获救人数Survived的平均值np.mean
passenger_survival = titanic_train.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)
print(passenger_survival)
#根据船舱等级Pclass计算年龄Age的平均值np.mean
passenger_age = titanic_train.pivot_table(index='Pclass', values='Age', aggfunc=np.mean)
print(passenger_age)
#根据登船港口Embarked计算票价Fare和存活率Survived的和np.sum
port_state = titanic_train.pivot_table(index='Embarked', values=['Fare','Survived'], aggfunc=np.sum)
print(port_state)
print('-------'*3)

#查看指定数据的值
row_83_Age = titanic_train.loc[83,'Age']#83行，Age列
print(row_83_Age)