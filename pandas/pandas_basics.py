import pandas as pd

food_info = pd.read_csv('food_info.csv')
#print(food_info.head())#默认显示前五行数据，或者指定行数3， food_info.head(3)
#print(food_info.tail())#默认显示后五行数据，或者指定行数3， food_info.tail(3)
columns_name = food_info.columns
print(columns_name)#获取csv列名
columns_name = columns_name.tolist() #将列名转换为list
print(columns_name)
print(food_info.shape)
print('-------'*3)

#print(food_info.loc[4]) #获取第四行的数据，按行获取数据，可以切片，包含两端
print('-------'*3)
#print(food_info[['NDB_No','Water_(g)']]) #按列获取数据,列名为一个列表，单独一个不用列表
print('-------'*3)
#从小到大排序，inplace=True在原表上修改, ascending=True默认为从小到大
food_info.sort_values('Water_(g)', inplace=True, ascending=True)
print(food_info['Water_(g)'])