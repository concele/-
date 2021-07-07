import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt

data=pd.read_csv('ml-1m/users.dat',sep='::',names=['user_id','item_id','rating','timestamp'])
# 拆分数据集并分别构建用户-物品矩阵
# 用户物品统计
n_users = data.user_id.nunique()
n_items = data.item_id.nunique()
from sklearn.model_selection import train_test_split
# 按照训练集70%，测试集30%的比例对数据进行拆分
train_data,test_data =train_test_split(data,test_size=0.3)
# 训练集 用户-物品 矩阵
train_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1,line[2]-1] = line[3]
# 测试集 用户-物品 矩阵
test_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1] = line[3]
# SVD矩阵

# 奇异值分解，超参数k的值就是设定要留下的特征值的数量
u, s, vt = svds(train_data_matrix,k=20)
s_diag_matrix = np.diag(s)
svd_prediction = np.dot(np.dot(u,s_diag_matrix),vt)

# 预测值限定最小值和最大值
# 预测值小于0的均设置为0，大于5的均设置为5
svd_prediction[svd_prediction < 0] =0
svd_prediction[svd_prediction > 5] =5


prediction_flatten = svd_prediction[train_data_matrix.nonzero()]
train_data_matrix_flatten = train_data_matrix[train_data_matrix.nonzero()]
error_train = sqrt(mean_squared_error(prediction_flatten,train_data_matrix_flatten))
print('训练集预测均方根误差：',error_train)

prediction_flatten = svd_prediction[test_data_matrix.nonzero()]
test_data_matrix_flatten = test_data_matrix[test_data_matrix.nonzero()]
error_test = sqrt(mean_squared_error(prediction_flatten,test_data_matrix_flatten))
print('测试集预测均方根误差：',error_test)