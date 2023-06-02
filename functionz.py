
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

#0-20  serie a 2020
#20-40  serie a 2019
#40-60  serie a 2018
#60-80  serie a 2017
#80-100 serie a 2016
#100-120 serie a 2015
#120-140  serie a 2014
'''data={"W":df.iloc[:,2],
      "D":df.iloc[:,3],
      "L":df.iloc[:,4],
      "G":df.iloc[:,5],
      "GA":df.iloc[:,6],
      "PTS":df.iloc[:,7],
      "xG":df.iloc[:,8],
      "NPxG":df.iloc[:,9],
      "xGA":df.iloc[:,10],
      "NPxGA":df.iloc[:,11],
      "NPxGD":df.iloc[:,12],
      "PPDA":df.iloc[:,13],
      "OPPDA":df.iloc[:,14],
      "DC":df.iloc[:,15],
      "ODC":df.iloc[:,16],
      "xPTS":df.iloc[:,17]
      }'''

def G(df):
    g= np.array([])
    g= np.append(g, df.iloc[:, 5])
    #assert g.size == 140
    return g

def GA(df):
    ga= np.array([])
    ga= np.append(ga, df.iloc[:, 6])
    #assert ga.size == 140
    return ga

def PTS(df):
    pts= np.array([])
    pts= np.append(pts, df.iloc[:, 7])
    #assert pts.size == 140
    return pts

def xG(df):
    xG=np.array([])
    xG=np.append(xG,df.iloc[:,8])
    #assert xG.size==140
    return xG

def NPxG(df):
    NPxG=np.array([])
    NPxG=np.append(NPxG,df.iloc[:,9])
    #assert NPxG.size==140
    return NPxG

def xGA(df):
    xGA= np.array([])

    xGA= np.append(xGA, df.iloc[:, 10])
    #assert xGA.size == 140
    return xGA

def NPxGA(df):
    NPxga= np.array([])
    NPxga= np.append(NPxga, df.iloc[:,11])
    #assert ga.size == 140
    return NPxga

def NPxGD(df):
    NPxgd= np.array([])
    NPxgd= np.append(NPxgd, df.iloc[:,12])
    #assert ga.size == 140
    return NPxgd

def PPDA(df):
    ppda= np.array([])

    ppda= np.append(ppda, df.iloc[:, 13])
    #assert ppda.size == 140
    return ppda

def OPPDA(df):
    oppda= np.array([])

    oppda= np.append(oppda, df.iloc[:, 14])
    #assert oppda.size == 140
    return oppda

def DC(df):
    dc= np.array([])
    dc= np.append(dc, df.iloc[:, 15])
    #assert dc.size == 140
    return dc

def ODC(df):
    odc= np.array([])

    odc= np.append(odc, df.iloc[:, 16])
    #assert odc.size == 140
    return odc

#Funzione generica per ottenere una colonna però bisogna specificare quale
def get_colonna(df,column_number):
    lol = np.array([])

    lol = np.append(lol, df.iloc[:, column_number])
    return lol

def champions_team(lista_df):
    squadre_champions=np.array([])
    i=0
    for df in lista_df:
        squadre_champions=np.append(squadre_champions,df.iloc[i:i+4,1:])
    return squadre_champions


def retrocesse(lista_df):
    retrocesse=np.array([])
    i=17
    for df in lista_df:
        retrocesse=np.append(retrocesse,df.iloc[i:i+3,1:])
    return retrocesse


#COME SONO FATTI I FILE JSON DENTRO
'''
"situation" :
"OpenPlay": "shots", "goals", "xG","against":"shots","goals","xG"
"FromCorner": uguale
"DirectFreekick":uguale
"SetPiece": uguale
"Penalty": uguale

------
"gameState":
"Goal diff 0":"time","shots","goals","xG","against":"shots","goals","xG"
"Goal diff +1":
"Goal diff > +1":
"Goal diff -1":
"Gaol diff < -1":

-----

"timing":
"1-15":"shots", "goals", "xG","against":"shots","goals","xG"
"16-30":
"31-45":
"46-60":
"61-75":
"76+":

----
"shotZone":
"ownGoals":"shots", "goals", "xG","against":"shots","goals","xG"
"shotOboxTotal":
"shotPenaltyArea":
"shotSixYardBox":

-----
"attackSpeed":
"Normal": "shots", "goals", "xG","against":"shots","goals","xG"
"Standard": 
"Slow":
"Fast":

-----
"result":
"SavedShot":"shots", "goals", "xG","against":"shots","goals","xG"
"MissedShots":
"BlockedShot":
"Goal":
"ShotOnPost":

'''
#FUNZIONI PER LA lista_stats_complete PER ACCEDERE A TUTTE LE FEATURES DEI FILE JSON E RESTITUIRE UNA COLONNA
def get_stats(lista_stats,campo1,campo2,campo3):
    x=np.array([])
    for i in range(0,140):
        x=np.append(x,lista_stats[i][campo1][campo2][campo3])
    return x

def get_stats_against(lista_stats,campo1,campo2,campo3):
    x = np.array([])
    for i in range(0, 140):
        x = np.append(x, lista_stats[i][campo1][campo2]["against"][campo3])
    return x

#FUNZIONE PER IL CLUSTER 
def RealClusterCreator(X,cl1,cr1):
    Xone = np.concatenate((X[cl1      :cr1, :] # 2020
    ,X[cl1 + 20 :cr1 + 20,  :]
  # 2019
    ,X[cl1 + 40 :cr1 + 40,  :]
  # ...
    ,X[cl1 + 60 :cr1 + 60,  :]

    ,X[cl1 + 80 :cr1 + 80,  :]

    ,X[cl1 + 100:cr1 + 100, :]

    ,X[cl1 + 120:cr1 + 120, :]),axis = 0)
    return(Xone)