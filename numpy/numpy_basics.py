import numpy as np

vector = np.array(['1','2','3'])
print(vector.dtype)
vector = vector.astype(float)#类型转换
print(vector.dtype)
print("*"*20)

matrix = np.array([[1,2,3],[4,9,6],[5,6,3]])
print(matrix.sum(axis=1))#按行求和
print(matrix.sum(axis=0))#按列求和
print("*"*20)

a = np.arange(0,15).reshape(3,5)#或者reshape(3,-1)  -1会自动根据情况计算出来
print(a)
b = a.ravel()#reshape的逆向操作
print(b)
print(a.shape)#数组大小
print(a.ndim)#数组维度
print(a.size)#数组中元素个数
print("*"*20)

zero_matrix = np.zeros((3,4))#零矩阵
print(zero_matrix)
ones_matrix = np.ones((2,3),dtype=np.int32)#全一矩阵
print(ones_matrix)
print("*"*20)

rand_matrix = np.random.random((3,5))
print(rand_matrix)
print("*"*20)

a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))
print(a)
print(b)
print(np.hstack((a,b)))#矩阵横着拼接
print(np.vstack((a,b)))#矩阵竖着拼接
print("*"*20)

c = np.floor(10*np.random.random((2,12)))
print(c)
print(np.hsplit(c,3))#横着切分，把C分成3份，竖着切为vsplit
print(np.hsplit(c,(3,5)))#横着切分，在指定位置进行切分（3,5）
print("*"*20)

a = np.arange(12).reshape(3,4)
b = a  #指向同一个对象
print(b is a)
b.shape = 2,6#修改形状会产生影响
print(a.shape)

c = a.view()  #不指向同一个对象，但是公用一套数值，浅复制
print(c is a)
a.shape = 1,12#修改形状不会产生影响
print(c.shape)
c[0,1] = 112323#修改数值会产生影响
print(a)

d = a.copy()#深复制，各自独立，常用这个
print("*"*20)

a = np.sin(np.arange(15).reshape(3,5))
print(a)
index = a.argmax(axis=0)#按列索引，找出每一列最大数值的索引
print(index)
value_max = a[index,range(a.shape[1])]#找出每一列最大数值
print(value_max)
print("*"*20)

a = np.arange(0,40,10).reshape(2,2)
print(a)
aa = np.tile(a,(2,3))#扩展操作
print(aa)
print("*"*20)

a = np.array([[2,5,1],[6,8,2]])
print(a)
b = np.sort(a,axis=1)#按行排序，从小到大
print(b)
print(a)
a.sort(axis=1)#直接对a进行排序
print(a)
print("-------------")
c = np.array([11,55,3])
d = np.argsort(c)#排序索引
print(d)
print(c[d])