import numpy as np


def loadExtData2():
    return np.mat(
            [[2, 0, 2, 0, 0, 4, 0, 1, 0, 0, 5],
           [1, 0, 0, 3, 0, 4, 0, 3, 0, 3, 3],
           [0, 0, 2, 0, 4, 0, 0, 3, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 2, 1, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 3, 0, 1, 2],
           [0, 2, 0, 0, 5, 0, 5, 0, 0, 4, 0],
           [1, 0, 0, 4, 0, 0, 0, 1, 2, 0, 0]])


def ecludSim(inA,inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))  #范数的计算方法linalg.norm()，这里的1/(1+距离)表示将相似度的范围放在0与1之间

def pearsSim(inA,inB):
    if len(inA)<3: return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)


def sigmaPct(sigma,percentage):
    sigma2=sigma**2
    sumsgm2=sum(sigma2)
    sumsgm3=0
    k=0
    for i in sigma:
        sumsgm3+=i**2
        k+=1
        if sumsgm3>=sumsgm2*percentage:
            return k


def svdEst(dataMat,user,simMeas,item,percentage):
    n=np.shape(dataMat)[1]
    simTotal=0.0;ratSimTotal=0.0
    u,sigma,vt=np.linalg.svd(dataMat)
    k=sigmaPct(sigma,percentage)
    sigmaK=np.mat(np.eye(k)*sigma[:k])
    xformedItems=dataMat.T*u[:,:k]*sigmaK.I
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T) #计算物品item与物品j之间的相似度
        simTotal+=similarity #对所有相似度求和
        ratSimTotal+=similarity*userRating #用"物品item和物品j的相似度"乘以"用户对物品j的评分"，并求和
    if simTotal==0:return 0
    else:return ratSimTotal/simTotal #得到对物品item的预测评分


def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=svdEst,percentage=0.9):
    dataMat=np.mat(dataMat)
    unratedItems=np.nonzero(dataMat[user,:].A==0)[1]  #建立一个用户未评分item的列表
    if len(unratedItems)==0:return 'you rated everything'
    itemScores=[]
    for item in unratedItems:  #对于每个未评分的item，都计算其预测评分
        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
        itemScores.append((item,estimatedScore))
    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)
    return itemScores[:N]

'''


a=recommend(data,-1,N=3,percentage=0.8)
data=np.mat(loadExtData2())
print(a[0][0])#对编号为1的用户推荐评分较高的3件商品

# 5
'''

