import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import numpy as np
import json
import functionz
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import accessodati as acv
import matplotlib.colors as mcolors
#INDIVIDUIAMO LE STATS DELLE SQUADRE CHAMPIONS E RETROCESSIONE
#Nota: (-1,17) impongo che il reshape avvenga a per forza 17 colonne e il numero di righe di conseguenza.
stats_champ_league=functionz.champions_team(acv.lista_dataframe_complete).reshape(-1,17)
stats_retrocesse=functionz.retrocesse(acv.lista_dataframe_complete).reshape(-1,17)

'''
#CREO DATAFRAME CON LE STATISTICHE PER LE SQUADRE CHAMPIONS
data_champ={"W":stats_champ_league[:,1],
      "D":stats_champ_league[:,2],
      "L":stats_champ_league[:,3],
      "G":stats_champ_league[:,4],
      "GA":stats_champ_league[:,5],
      "PTS":stats_champ_league[:,6],
      "xG":stats_champ_league[:,7],
      "NPxG":stats_champ_league[:,8],
      "xGA":stats_champ_league[:,9],
      "NPxGA":stats_champ_league[:,10],
      "NPxGD":stats_champ_league[:,11],
      "PPDA":stats_champ_league[:,12],
      "OPPDA":stats_champ_league[:,13],
      "DC":stats_champ_league[:,14],
      "ODC":stats_champ_league[:,15],
      "xPTS":stats_champ_league[:,16]
      }
dataframe_champions=pd.DataFrame(data=data_champ,columns=["W","D","L","G", "GA", "PTS", "xG", "NPxG", "xGA", "NPxGA", "NPxGD", "PPDA", "OPPDA", "DC", "ODC", "xPTS"])
#print(dataframe_champions)

#CREO DATAFRAME CON LE SQUADRE RETROCESSE
data_retro={"W":stats_retrocesse[:,1],
      "D":stats_retrocesse[:,2],
      "L":stats_retrocesse[:,3],
      "G":stats_retrocesse[:,4],
      "GA":stats_retrocesse[:,5],
      "PTS":stats_retrocesse[:,6],
      "xG":stats_retrocesse[:,7],
      "NPxG":stats_retrocesse[:,8],
      "xGA":stats_retrocesse[:,9],
      "NPxGA":stats_retrocesse[:,10],
      "NPxGD":stats_retrocesse[:,11],
      "PPDA":stats_retrocesse[:,12],
      "OPPDA":stats_retrocesse[:,13],
      "DC":stats_retrocesse[:,14],
      "ODC":stats_retrocesse[:,15],
      "xPTS":stats_retrocesse[:,16]
      }
dataframe_retro=pd.DataFrame(data=data_retro,columns=["W","D","L","G", "GA", "PTS", "xG", "NPxG", "xGA", "NPxGA", "NPxGD", "PPDA", "OPPDA", "DC", "ODC", "xPTS"])
#print(dataframe_retro)
'''
import seaborn as sns

X = acv.X_SUPREMO
predictors=["G","GA","PPDA","OPPDA","DC","ODC"]
#fig, ax = plt.subplots(dim/2,2)

cl1 = 0    # cluster 1 limite sinistro
cr1 = 4    # cluster 1 limite destro

cl2 = 5    # cluster 2 limite sinistro
cr2 = 7    # cluster 2 limite destro

clb = 10   # cluster a caso limite sinistro
crb = 15   # cluster a caso limite destro

cl3 = 17   # cluster 3 limite sinistro
cr3 = 20   # cluster 3 limite destro

cl1_col = 'gold'       # cluster 1 colore
cl2_col = 'firebrick'  # cluster 2 colore
cl3_col = 'green'      # cluster 3 colore
clb_col = 'teal'       # cluster a caso colore

X_cl1 = functionz.RealClusterCreator(X,cl1, cr1)
X_cl2 = functionz.RealClusterCreator(X,cl2, cr2)
X_cl3 = functionz.RealClusterCreator(X,cl3, cr3)
X_cl4 = functionz.RealClusterCreator(X,clb, crb)

label1 = np.ones_like(range(0,np.shape(X_cl1)[0]))*1
label2 = np.ones_like(range(0,np.shape(X_cl2)[0]))*2
label3 = np.ones_like(range(0,np.shape(X_cl3)[0]))*3
label4 = np.ones_like(range(0,np.shape(X_cl4)[0]))*4

label1 = label1.reshape(-1,1)
label2 = label2.reshape(-1,1)
label3 = label3.reshape(-1,1)
label4 = label4.reshape(-1,1)


X = np.concatenate((X_cl1,X_cl2,X_cl3,X_cl4),axis=0)
Y = np.concatenate((label1,label2,label3,label4),axis=0)

T = np.concatenate((X,Y),axis = 1)

df = pd.DataFrame(T)
df.columns = acv.COLONNE + ['label']
df = df[[acv.COLONNE[0],acv.COLONNE[1],acv.COLONNE[2], acv.COLONNE[3],acv.COLONNE[4], acv.COLONNE[5],'label']]
print(df)

#PLOT
df.columns = ["G","GA","PPDA","OPPDA","DC","ODC", 'label']
colori = list(mcolors.TABLEAU_COLORS.values())
col_dict = {1:colori[0], 2:colori[1] , 3:colori[2], 4:colori[3]}
eva = sns.PairGrid(df, hue='label',palette=col_dict)
eva.map_diag(sns.histplot)
eva.map_offdiag(sns.scatterplot)
eva.add_legend()
#print(np.shape(X)[0])
plt.show()


