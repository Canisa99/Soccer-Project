import matplotlib.pyplot as plt
import accessodati as acv


# crea cluster, considera le squadre dal cl1-esimo al cr1-esimo posto per tutte le stagioni,
# cl1_col è il colore da visualizzare, mark il marcatore, lab l'etichetta
def ClusterCreator(cl1,cr1,cl1_col,mark,lab):
    return(plt.scatter(acv.X_six_std[cl1      :cr1, i], acv.pts_s[cl1      :cr1      ], color=cl1_col,marker = mark,label = lab) # 2020
    ,plt.scatter(acv.X_six_std[cl1 + 20 :cr1 + 20,  i], acv.pts_s[cl1 + 20 :cr1 +  20], color=cl1_col,marker = mark)  # 2019
    ,plt.scatter(acv.X_six_std[cl1 + 40 :cr1 + 40,  i], acv.pts_s[cl1 + 40 :cr1 +  40], color=cl1_col,marker = mark)  # ...
    ,plt.scatter(acv.X_six_std[cl1 + 60 :cr1 + 60,  i], acv.pts_s[cl1 + 60 :cr1 +  60], color=cl1_col,marker = mark)
    ,plt.scatter(acv.X_six_std[cl1 + 80 :cr1 + 80,  i], acv.pts_s[cl1 + 80 :cr1 +  80], color=cl1_col,marker = mark)
    ,plt.scatter(acv.X_six_std[cl1 + 100:cr1 + 100, i], acv.pts_s[cl1 + 100:cr1 + 100], color=cl1_col,marker = mark)
    ,plt.scatter(acv.X_six_std[cl1 + 120:cr1 + 120, i], acv.pts_s[cl1 + 120:cr1 + 120], color=cl1_col,marker = mark))


#regressione lineare: PTS target
#G,GA,PPDA,OPPDA,DC,ODC  predictors
predictors=["G","GA","PPDA","OPPDA","DC","ODC"]
#fig, ax = plt.subplots(dim/2,2)

cl1 = 0   # cluster 1 limite sinistro
cr1 = 4   # cluster 1 limite destro

cl2 = 4   # cluster 2 limite sinistro
cr2 = 7   # cluster 2 limite destro

cl3 = 17   # cluster 3 limite sinistro
cr3 = 20   # cluster 3 limite destro

cl1_col = 'gold'       # cluster 1 colore
cl2_col = 'firebrick'  # cluster 2 colore
cl3_col = 'green'      # cluster 3 colore

for i in range(len(predictors)):
    plt.figure(i)
    #plt.scatter(regressioni.X[:,i], regressioni.pts_s,color='navy', marker = '+')

    ClusterCreator(cl1, cr1, cl1_col, '*', 'Champions')
    ClusterCreator(cl2, cr2, cl2_col, 'p', 'Europa')
    ClusterCreator(cl3, cr3, cl3_col, '1', 'Retrocessione')

    plt.title("PTS vs " + predictors[i])
    plt.ylabel("PTS")
    plt.xlabel(predictors[i])
    plt.legend()


plt.tight_layout()
plt.show()