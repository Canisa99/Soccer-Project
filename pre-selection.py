import accessodati as acv
from scipy.stats import pearsonr
import numpy as np

X= np.concatenate((acv.X_six_std,acv.X_more_std),axis=1)
X=np.delete(X,[0,1,acv.COLONNE.index("res_goal_shots")],axis=1)
y=acv.pts_s
matrix=np.array([])
index_feat=[]
lista_corr=[]
for i in range(0,X.shape[1]):
    for j in range(0,X.shape[1]):
        corr,_=pearsonr(X[:,i],X[:,j])
        if (np.abs(corr) >0.90 and np.abs(corr)<0.99):
            if (j,i) not in index_feat:
                    lista_corr.append(corr)
                    index_feat.append((i,j))

#INDICI delle coppie altamente correlate tra di loro
print(index_feat)
print(len(index_feat))
print(lista_corr)
#Ci prendiamo nella coppia solo quella più correlata alla variabile target, l'altra la scartiamo
sel_coppie=[]
var_scartate=[]
for i in index_feat:
    corr0,_=pearsonr(X[:,i[0]],y[:,0])
    corr1,_=pearsonr(X[:,i[1]],y[:,0])
    if np.abs(corr0)>np.abs(corr1):
        sel_coppie.append(i[0])
        var_scartate.append(i[1])
    else:
        sel_coppie.append(i[1])
        var_scartate.append(i[0])
print(sel_coppie)
print(var_scartate)

#Rimuovo duplicati
resultantList = []
for element in var_scartate:
    if element not in resultantList:
        resultantList.append(element)
print(resultantList)
print(len(resultantList))
#Traslo di 1 perchè R parte da 1 e non da 0
lista=np.array(resultantList)
lista=lista+1
print(lista)

import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))
adjr2=[0.7847737,0.9531876,0.9552628,0.9549594,0.9546288,0.9542877]
asse_x=[1,2,3,4,5,6]
plt.plot(asse_x, adjr2,color='lightblue', marker='o', linestyle='-',
     linewidth=2, markersize=10)
plt.plot([3],[0.9552628],color="red",marker="o",markersize=16)
plt.xlabel('Number of Features',size=12)
plt.ylabel('Adjusted $R^2$',size=12)
plt.title("Best Subset Selection",size=18)
#plt.legend(['test score', 'train score'], loc='upper left')
#plt.show()
