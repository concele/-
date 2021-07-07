import numpy as np
from web.load_data import load_user_data
'''
u,r=load_user_data()
r=np.mat(r)

cf_matrix=np.zeros((6040,3952))
for i in range(r.shape[0]):
    matrix=r[i]
    cf_matrix[matrix[0,0]-1][matrix[0,1]-1] = matrix[0,2]
    
cf=np.load('cf_matrix.npy')
cf_matrix2=np.zeros((200,200))
for i in range(0,200):
  for j in range(0,200):
    cf_matrix2[i][j]=cf[i][j]
    
np.save('cf.npy',cf_matrix2)
n=np.load('cf.npy')
print(n,n.shape)
'''


