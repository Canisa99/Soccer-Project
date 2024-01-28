import pandas as pd
import numpy as np
import functionz
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import csv
import numpy as np
import pandas as pd


############################
###### PREMIER LEAGUE ######
###########################


#DATAFRAME PORTIERI PREMIER LEAGUE
wagespl=pd.read_csv("WAGESPL.txt",delimiter=";")
#print(wagespl)
stats_gk_pl=pd.read_csv("Stats_portieri_premier.txt")
stats_adv_gk_pl=pd.read_csv("Stats_portieri_adv_premier.txt")
#14 saves, 19 cs, 15 psxg
data_gk_pl={"Player":stats_gk_pl.iloc[:,1],
         "Saves":stats_gk_pl.iloc[:,14],
         "Clean sheets":stats_gk_pl.iloc[:,19],
         "PSxG":stats_adv_gk_pl.iloc[:,15]}
stats_complete_gk_pl=pd.DataFrame(data=data_gk_pl,columns=["Player","Saves","Clean sheets","PSxG"])
df_gk_pl=stats_complete_gk_pl.merge(wagespl,on="Player")
#print(df_gk)

#DATAFRAME GENERALE PREMIER LEAGUE
wages_premier=pd.read_csv("WAGESPL.txt",delimiter=";")
#print(wages_premier)
stats_giocatori_premier=pd.read_csv("JSONplPL.txt")
#4=goals 6=assists 9=keypasses 16=chain 17=buildup
data_giocatori_premier={"Player":stats_giocatori_premier.iloc[:,1],
         "Goals":stats_giocatori_premier.iloc[:,4],
         "Assists":stats_giocatori_premier.iloc[:,6],
         "Keypasses":stats_giocatori_premier.iloc[:,9],
         "xGChain":stats_giocatori_premier.iloc[:,16],
         "xGBuildup":stats_giocatori_premier.iloc[:,17]}
stats_complete_giocatori_premier=pd.DataFrame(data=data_giocatori_premier,columns=["Player","Goals","Assists","Keypasses","xGChain","xGBuildup"])
df_giocatori_premier=stats_complete_giocatori_premier.merge(wages_premier,on="Player")
#print(df_giocatori_premier)


#DATAFRAME DIFENSORI PREMIER
name_defender_premier=np.array([])
g_defender_premier=np.array([])
a_defender_premier=np.array([])
kp_defender_premier=np.array([])
xgchain_defender_premier=np.array([])
xgbuildup_defender_premier=np.array([])
dataframe_premier=pd.read_csv("JSONplPL.txt")
for i in range(dataframe_premier.shape[0]):
    if "D" == dataframe_premier.iloc[i, 12] or "D M" == dataframe_premier.iloc[i, 12] or "D M S" == dataframe_premier.iloc[i, 12] or "D S" ==dataframe_premier.iloc[i, 12]:
        name_defender_premier=np.append(name_defender_premier,dataframe_premier.iloc[i,1])
        g_defender_premier=np.append(g_defender_premier,dataframe_premier.iloc[i,4])
        a_defender_premier=np.append(a_defender_premier,dataframe_premier.iloc[i,6])
        kp_defender_premier=np.append(kp_defender_premier,dataframe_premier.iloc[i,9])
        xgchain_defender_premier=np.append(xgchain_defender_premier,dataframe_premier.iloc[i,16])
        xgbuildup_defender_premier=np.append(xgbuildup_defender_premier,dataframe_premier.iloc[i,17])
data_defender_premier={"Player":name_defender_premier,"Gol":g_defender_premier,"Assist":a_defender_premier,"Key passes":kp_defender_premier,"xGchain":xgchain_defender_premier,"xGbuildup":xgbuildup_defender_premier}
dataframe_defender_premier=pd.DataFrame(data=data_defender_premier,columns=["Player","Gol","Assist","Key passes","xGchain","xGbuildup"])
df_defender_premier=dataframe_defender_premier.merge(wages_premier,on="Player")
df_defender_premier=df_defender_premier.drop(columns=['Position'])
#print(df_defender_premier)

#CENTROCAMPISTI PREMIER
name_pl = np.array([])
g_middle_pl = np.array([])
a_middle_pl = np.array([])
kp_middle_pl = np.array([])
xgchain_middle_pl = np.array([])
xgbuildup_middle_pl = np.array([])
dataframe_pl=pd.read_csv("JSONplPL.txt")
for i in range(dataframe_pl.shape[0]):
    if "M"==dataframe_pl.iloc[i,12] or "M S"==dataframe_pl.iloc[i,12]:
        name_pl = np.append(name_pl,dataframe_pl.iloc[i,1])
        g_middle_pl = np.append(g_middle_pl , dataframe_pl.iloc[i,4])
        a_middle_pl = np.append(a_middle_pl , dataframe_pl.iloc[i,6])
        kp_middle_pl = np.append(kp_middle_pl , dataframe_pl.iloc[i,9])
        xgchain_middle_pl = np.append(xgchain_middle_pl , dataframe_pl.iloc[i,16])
        xgbuildup_middle_pl = np.append(xgbuildup_middle_pl , dataframe_pl.iloc[i,17])

data_middle_pl = {"Player":name_pl,"Gol":g_middle_pl,"Assist":a_middle_pl,"Key passes":kp_middle_pl,
                                  "xGchain":xgchain_middle_pl,"xGbuildup":xgbuildup_middle_pl}
dataframe_middle_pl = pd.DataFrame(data=data_middle_pl,columns=["Player","Gol","Assist","Key passes","xGchain","xGbuildup"])
df_m_pl = dataframe_middle_pl.merge(wages_premier,on="Player")
df_m_pl=df_m_pl.drop(columns=['Position'])
#print(df_m_pl)

#ATTACCANTI PREMIER
name_forward_premier=np.array([])
g_forward_premier=np.array([])
a_forward_premier=np.array([])
kp_forward_premier=np.array([])
xgchain_forward_premier=np.array([])
xgbuildup_forward_premier=np.array([])
for i in range(dataframe_pl.shape[0]):
    if "F"==dataframe_pl.iloc[i,12] or "F S"==dataframe_pl.iloc[i,12]:
        name_forward_premier=np.append(name_forward_premier,dataframe_pl.iloc[i,1])
        g_forward_premier=np.append(g_forward_premier,dataframe_pl.iloc[i,4])
        a_forward_premier=np.append(a_forward_premier,dataframe_pl.iloc[i,6])
        kp_forward_premier=np.append(kp_forward_premier,dataframe_pl.iloc[i,9])
        xgchain_forward_premier=np.append(xgchain_forward_premier,dataframe_pl.iloc[i,16])
        xgbuildup_forward_premier=np.append(xgbuildup_forward_premier,dataframe_pl.iloc[i,17])

data_forward_premier={"Player":name_forward_premier,"Gol":g_forward_premier,"Assist":a_forward_premier,"Key passes":kp_forward_premier,"xGchain":xgchain_forward_premier,"xGbuildup":xgbuildup_forward_premier}
dataframe_forward_premier=pd.DataFrame(data=data_forward_premier,columns=["Player","Gol","Assist","Key passes","xGchain","xGbuildup"])
df_forward_premier=dataframe_forward_premier.merge(wages_premier,on="Player")
df_forward_premier=df_forward_premier.drop(columns=['Position'])
#print(df_forward_premier)

#########################
###### SERIE A #########
#######################

#DATAFRAME GENERALE SERIE A
wages_seriea=pd.read_csv("WAGESSERIEA.txt",delimiter=";")
#print(wages_seriea)
stats_giocatori_seriea=pd.read_csv("JSONplA.txt")
#4=goals 6=assists 9=keypasses 16=chain 17=buildup
data_giocatori_seriea={"Player":stats_giocatori_seriea.iloc[:,1],
         "Goals":stats_giocatori_seriea.iloc[:,4],
         "Assists":stats_giocatori_seriea.iloc[:,6],
         "Keypasses":stats_giocatori_seriea.iloc[:,9],
         "xGChain":stats_giocatori_seriea.iloc[:,16],
         "xGBuildup":stats_giocatori_seriea.iloc[:,17]}
stats_complete_giocatori_seriea=pd.DataFrame(data=data_giocatori_seriea,columns=["Player","Goals","Assists","Keypasses","xGChain","xGBuildup"])
df_giocatori_seriea=stats_complete_giocatori_seriea.merge(wages_seriea,on="Player")
#print(df_giocatori_seriea)

dataframe=pd.read_csv("JSONplA.txt")

#DATAFRAME DIFENSORI SERIE A
name_defender_seriea=np.array([])
g_defender_seriea=np.array([])
a_defender_seriea=np.array([])
kp_defender_seriea=np.array([])
xgchain_defender_seriea=np.array([])
xgbuildup_defender_seriea=np.array([])
for i in range(dataframe.shape[0]):
    if "D"==dataframe.iloc[i,12] or "D M"==dataframe.iloc[i,12] or "D M S"==dataframe.iloc[i,12] or "D S"==dataframe.iloc[i,12]:
        name_defender_seriea=np.append(name_defender_seriea,dataframe.iloc[i,1])
        g_defender_seriea=np.append(g_defender_seriea,dataframe.iloc[i,4])
        a_defender_seriea=np.append(a_defender_seriea,dataframe.iloc[i,6])
        kp_defender_seriea=np.append(kp_defender_seriea,dataframe.iloc[i,9])
        xgchain_defender_seriea=np.append(xgchain_defender_seriea,dataframe.iloc[i,16])
        xgbuildup_defender_seriea=np.append(xgbuildup_defender_seriea,dataframe.iloc[i,17])
data_defender={"Player":name_defender_seriea,"Gol":g_defender_seriea,"Assist":a_defender_seriea,"Key passes":kp_defender_seriea,"xGchain":xgchain_defender_seriea,"xGbuildup":xgbuildup_defender_seriea}
dataframe_defender_seriea=pd.DataFrame(data=data_defender,columns=["Player","Gol","Assist","Key passes","xGchain","xGbuildup"])
df_defender_seriea=dataframe_defender_seriea.merge(wages_seriea,on="Player")
df_defender_seriea=df_defender_seriea.drop(columns=['Position'])

#DATAFRAME CENTROCAMPISTI SERIE A
name_middle_seriea=np.array([])
g_middle_seriea = np.array([])
a_middle_seriea = np.array([])
kp_middle_seriea = np.array([])
xgchain_middle_seriea = np.array([])
xgbuildup_middle_seriea = np.array([])
for i in range(dataframe.shape[0]):
    if "M"==dataframe.iloc[i,12] or "M S"==dataframe.iloc[i,12]:
        name_middle_seriea = np.append(name_middle_seriea,dataframe.iloc[i,1])
        g_middle_seriea = np.append(g_middle_seriea , dataframe.iloc[i,4])
        a_middle_seriea = np.append(a_middle_seriea , dataframe.iloc[i,6])
        kp_middle_seriea = np.append(kp_middle_seriea , dataframe.iloc[i,9])
        xgchain_middle_seriea = np.append(xgchain_middle_seriea , dataframe.iloc[i,16])
        xgbuildup_middle_seriea = np.append(xgbuildup_middle_seriea , dataframe.iloc[i,17])

data_middle_seriea = {"Player":name_middle_seriea,"Gol":g_middle_seriea,"Assist":a_middle_seriea,"Key passes":kp_middle_seriea,
                                  "xGchain":xgchain_middle_seriea,"xGbuildup":xgbuildup_middle_seriea}
dataframe_middle_seriea = pd.DataFrame(data=data_middle_seriea,columns=["Player","Gol","Assist","Key passes","xGchain","xGbuildup"])
df_m = dataframe_middle_seriea.merge(wages_seriea,on="Player")
df_m=df_m.drop(columns=['Position'])
#print(df_m)

#DATAFRAME ATTACCANTI SERIE A
name_forward_seriea=np.array([])
g_forward_seriea=np.array([])
a_forward_seriea=np.array([])
kp_forward_seriea=np.array([])
xgchain_forward_seriea=np.array([])
xgbuildup_forward_seriea=np.array([])
for i in range(dataframe.shape[0]):
    if "F"==dataframe.iloc[i,12] or "F S"==dataframe.iloc[i,12]:
        name_forward_seriea=np.append(name_forward_seriea,dataframe.iloc[i,1])
        g_forward_seriea=np.append(g_forward_seriea,dataframe.iloc[i,4])
        a_forward_seriea=np.append(a_forward_seriea,dataframe.iloc[i,6])
        kp_forward_seriea=np.append(kp_forward_seriea,dataframe.iloc[i,9])
        xgchain_forward_seriea=np.append(xgchain_forward_seriea,dataframe.iloc[i,16])
        xgbuildup_forward_seriea=np.append(xgbuildup_forward_seriea,dataframe.iloc[i,17])

data_forward={"Player":name_forward_seriea,"Gol":g_forward_seriea,"Assist":a_forward_seriea,"Key passes":kp_forward_seriea,"xGchain":xgchain_forward_seriea,"xGbuildup":xgbuildup_forward_seriea}
dataframe_forward=pd.DataFrame(data=data_forward,columns=["Player","Gol","Assist","Key passes","xGchain","xGbuildup"])
df_forward_seriea=dataframe_forward.merge(wages_seriea,on="Player")
df_forward_seriea=df_forward_seriea.drop(columns=['Position'])
#print(df_forward_seriea)





###################################
######SUPPORT VECTOR MACHINE######
##################################
#CLASSIFICHIAMO IN BASE ALLE STATISTICHE SE UN GIOCATORE HA STIPEDIO ALTO O BASSO
"""
lista_dataframe_seriea=[df_defender_seriea,df_m,df_forward_seriea]
lista_dataframe_pl=[df_defender_premier,df_m_pl,df_forward_premier]
c=0
for i,j in zip(lista_dataframe_seriea,lista_dataframe_pl):
    if c==0:
        print("#########")
        print("DIFENSORI:")
        print("#########")
    if c==1:
        print("#########")
        print("CENTROCAMPISTI")
        print("#########")
    if c==2:
        print("#########")
        print("ATTACCANTI")
        print("#########")
    # MEDIANA CENTR SERIEA
    median_salary = np.median(np.array(i.iloc[:, 6]))
    #print(median_salary)

    # MEDIANA CENTR PREMIER
    median_salary_pl = np.median(np.array(j.iloc[:, 6]))
    #print(median_salary_pl)

    target = np.where(i.iloc[:, 6] > median_salary, 1, 0)
    # print(target.reshape(-1,1))
    i['Target'] = target.reshape(-1, 1)
    # print(df_gk.columns)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf = svm.SVC()
    Gol = np.array(i["Gol"]).reshape(-1, 1)
    Assist = np.array(i["Assist"]).reshape(-1, 1)
    KeyPasses = np.array([i["Key passes"]]).reshape(-1, 1)
    xGchain = np.array([i['xGchain']]).reshape(-1, 1)
    xGbuild = np.array([i['xGbuildup']]).reshape(-1, 1)
    age = np.array(i['Age']).reshape(-1, 1)

    X = np.concatenate((Gol, Assist, KeyPasses, xGchain, xGbuild, age), axis=1)
    # print(X)
    scaler = StandardScaler()
    scaler.fit_transform(X)
    grid = dict()
    grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # grid['decision_function_shape'] = ['ovo','ovr']
    # define search
    search = GridSearchCV(clf, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X, target)
    # summarize
    if c==0:
        print("Difensori Serie A:")
    if c==1:
        print("Centrocampisti Serie A:")
    if c==2:
        print("Attaccanti Serie A:")
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)

    target_pl = np.where(j.iloc[:, 6] > median_salary_pl, 1, 0)
    # print(target.reshape(-1,1))
    j['Target'] = target_pl.reshape(-1, 1)
    # print(df_gk.columns)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf = svm.SVC()
    Gol_pl = np.array(j["Gol"]).reshape(-1, 1)
    Assist_pl = np.array(j["Assist"]).reshape(-1, 1)
    KeyPasses_pl = np.array([j["Key passes"]]).reshape(-1, 1)
    xGchain_pl = np.array([j['xGchain']]).reshape(-1, 1)
    xGbuild_pl = np.array([j['xGbuildup']]).reshape(-1, 1)
    age_pl = np.array(j['Age']).reshape(-1, 1)

    X_pl = np.concatenate((Gol_pl, Assist_pl, KeyPasses_pl, xGchain_pl, xGbuild_pl, age_pl), axis=1)
    # print(X)
    scaler = StandardScaler()
    scaler.fit_transform(X_pl)

    grid = dict()
    grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # grid['decision_function_shape'] = ['ovo','ovr']
    # define search
    search = GridSearchCV(clf, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X_pl, target_pl)
    # summarize
    if c==0:
        print("Difensori Premier League:")
    if c==1:
        print("Centrocampisti Premier League:")
    if c==2:
        print("Attaccanti Premier League:")
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)

    X_supremo = np.concatenate((X, X_pl), axis=0)
    target_supremo = np.concatenate((target, target_pl), axis=0)

    scaler = StandardScaler()
    scaler.fit_transform(X_supremo)

    grid = dict()
    grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # grid['decision_function_shape'] = ['ovo','ovr']
    # define search
    search = GridSearchCV(clf, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X_supremo, target_supremo)
    # summarize
    if c==0:
        print("Difensori da entrambe:")
    if c==1:
        print("Centrocampisti da entrambe:")
    if c==2:
        print("Attaccanti da entrambe:")
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    c=c+1
"""