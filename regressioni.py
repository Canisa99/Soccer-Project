import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import numpy as np
import json
import functionz
import accessodati as acv
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

#INDIVIDUIAMO LE STATS DELLE SQUADRE CHAMPIONS E RETROCESSIONE
stats_champ_league=acv.stats_champ_league
stats_retrocesse=acv.stats_retrocesse

dataframe_definitivo=acv.dataframe_definitivo

###PROBLEMA CON TUTTE LE 157 FEATURES###
"""
#DEFINISCO X e Y
X_SUPREMO=acv.X_SUPREMO_std
y_SUPREMO=functionz.PTS(dataframe_definitivo).reshape(-1,1)

#SPLITTING
X_supremo_train, X_supremo_test, y_supremo_train, y_supremo_test = train_test_split(X_SUPREMO,y_SUPREMO,test_size=0.2,random_state=1)

#STANDARDIZZO X
sc_x = StandardScaler()
sc_x.fit(X_supremo_train)
X_supremo_train_std = sc_x.transform(X_supremo_train)
X_supremo_test_std = sc_x.transform(X_supremo_test)

# STANDARDIZZO Y
sc_y= StandardScaler()
sc_y.fit(y_supremo_train)
y_supremo_train_std = sc_y.transform(y_supremo_train)
y_supremo_test_std = sc_y.transform(y_supremo_test)

lm = LinearRegression()
lm.fit(X_supremo_train_std,y_supremo_train_std)
y_supremo_pred=lm.predict(X_supremo_test_std)
r2 = sklearn.metrics.r2_score(y_supremo_test_std, y_supremo_pred)
print(r2)
model_1 = sm.OLS(y_supremo_train_std, sm.add_constant(X_supremo_train_std)).fit()
print(model_1.summary())
"""

###PTS as target and G,GA,PPDA,OPPDA,DC,ODC as predictors
X=acv.X_six_std
y=acv.pts_s
#dimesione 140x6

#SPLITTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=sm.OLS(y_train,X_train).fit()
print(model.summary())
y_pred = model.predict(X_test)
r2_test = sklearn.metrics.r2_score(y_test, y_pred)
print(r2_test)



#Forward Stepwise Selection

#Aggiunge a mano a mano feature solo se il p-value associato alla feature appena aggiunta nella regressione è inferiore a 0.05
#se è superiore la variabile viene rimossa

columns=['g','ga','ppda','oppda','dc','odc']
df_pred_train=pd.DataFrame(data=X_train,columns=columns)

count_predictors=[]  #ad ogni passo della forward selection tiene conto del numero di predittori considerati
bic_stats=[]   #contiene BIC per ogni passo della forward selection
sel_features=[]   # alla fine del ciclo conterrà le variabili selezionate tali che il p-value associato sia <0.05
for i in range(len(columns)):
    sel_features.append(columns[i])
    #print(sel_features)
    #ALLENO IL MODELLO PRENDENDO IN CONSIDERAZIONE SOLO LE COLONNE DELLA LISTA CHE SELEZIONA LE COLONNE
    model=sm.OLS(y_train,df_pred_train[sel_features]).fit()
    count_predictors.append(len(sel_features))
    bic_stats.append(model.bic)
    #SE L'ULTIMA VARIABILE AGGIUNTA HA UN PVALUE MAGGIORE DI 0.05 ALLORA VIENE CANCELLATA
    if model.pvalues[-1]>0.05:
        del sel_features[-1]
#DOPO LA SELEZIONI POSSO FARE LA REGRESSIONE CON LE FEATURES SELEZIONATE
model=sm.OLS(y_train,df_pred_train[sel_features]).fit()
#print(model.summary())
#ATTENZIONE: STIAMO SOLO LAVORANDO SUL TRAINING, A NOI INTERESSA LA PREDIZIONE SUL TEST E DOBBIAMO CONSIDERARLO, PERCIò
#USIAMO LA CROSS VALIDATION

###FEATURE SELECTION UTILIZZANDO K-FOLD CROSS VALIDATION
#Hyperparameter Tuning Using Grid Search Cross-Validation
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


#print(len(columns))
#KFOLD DIVIDE IL DATASET IN n_splits CARTELLE (NEL NOSTRO CASO 4)
folds = KFold(n_splits = 4, shuffle = True, random_state = 100)
#PARAMETRO NECESSARIO A GRIDSEARCH (è UN DIZIONARIO)
hyper_params = [{'n_features_to_select': list(range(1, len(columns)+1))}]

lm = LinearRegression()
lm.fit(X_train, y_train)

#RFE
#To build the model using RFE, we need to tell RFE how many features we want in the final model.
#It then runs a feature elimination algorithm.
"""

#ALGORITMO DI SELEZIONE DELLE VARIABILI
rfe = RFE(lm)

#SELEZIONATORE BASATO SU UNA REGRESSIONE LINERARE
model_cv = GridSearchCV(estimator = rfe,
                        param_grid = hyper_params,
                        scoring= 'neg_mean_squared_error', #CRITERIO CON CUI MI FA LA SELEZIONE
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

# fit the model
model_cv.fit(X_train, y_train)
#RISULTATI
cv_results = pd.DataFrame(model_cv.cv_results_)

#PLOTTIAMOOO

plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('neg_mean_squared_error')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')
plt.show()

#DOPO AVER VISTO IL GRAFICO CONCLUDIAMO CHE:
n_features_optimal = 3

lm = LinearRegression()
lm.fit(X_train, y_train)

#FAMMI LA SELEZIONE DELLE VARIABILI COL VINCOLO CHE SIANO SOLO n_features_optimal
rfe = RFE(lm, n_features_to_select=n_features_optimal)
rfe = rfe.fit(X_train, y_train)

#capire l'indice(rispetto a X) delle variabili selezionate cioè quali effettivamente sono le n_features_to_select migliori
bool_feat_sel=list(rfe.get_support())
index_feat_sel=[i for i in range(len(bool_feat_sel)) if bool_feat_sel[i]==True]
#print(index_feat_sel) #restituisce le variabili 0,1,3
#print(rfe.get_support())

y_pred = rfe.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)

#reduce X to the selected features
X=rfe.transform(X)
#print(X)


'''
#creo dataframe dei predittori

#columns=['g','ga','ppda','oppda','dc','odc']
#df_pred=pd.DataFrame(data=X,columns=columns)
#print(df_pred)


#testing collinearity
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_pred.values, i) for i in range(df_pred.shape[1])]
vif["features"] = ['g','ga','ppda','oppda','dc','odc']
print(vif)
#     VIF Factor features
# 0    3.535563        g
# 1    2.494024       ga
# 2    1.619333     ppda
# 3    2.729200    oppda
# 4    3.925628       dc
# 5    3.467075      odc
'''


#trovare outliers tramite residui studentizzati (valore maggiore di 3 indica outlier) per migliorare il fit lineare


from statsmodels.formula.api import ols
#STAVOLTA USIAMO LE X SELEZIONATE CHE SONO 3
model=sm.OLS(pts_s,X).fit()
print(model.summary())
#print(model.pvalues)
adjusted_Rsq=np.array([model.rsquared_adj])
influence = model.get_influence()
studentized_residuals = influence.resid_studentized_external
#print(studentized_residuals)
index_outliers=[i for i in range(len(studentized_residuals)) if studentized_residuals[i]>=3 or studentized_residuals[i]<=-3]
nobs_per_iteration=np.array([140])
#print(index_outliers)
"""
"""
while len(index_outliers)!=0:


    dataframe_definitivo=dataframe_definitivo.drop(index=index_outliers,axis=0)
    dataframe_definitivo = dataframe_definitivo.reset_index(drop=True)
    g = functionz.G(dataframe_definitivo).reshape(-1, 1)
    ga = functionz.GA(dataframe_definitivo).reshape(-1, 1)
    ppda = functionz.PPDA(dataframe_definitivo).reshape(-1, 1)
    oppda = functionz.OPPDA(dataframe_definitivo).reshape(-1, 1)
    dc = functionz.DC(dataframe_definitivo).reshape(-1, 1)
    odc = functionz.ODC(dataframe_definitivo).reshape(-1, 1)
    pts = functionz.PTS(dataframe_definitivo).reshape(-1, 1)

    # scaling features
    g_s = (g - np.mean(g)) / np.std(g)
    ga_s = (ga - np.mean(ga)) / np.std(ga)
    ppda_s = (ppda - np.mean(ppda)) / np.std(ppda)
    oppda_s = (oppda - np.mean(oppda)) / np.std(oppda)
    dc_s = (dc - np.mean(dc)) / np.std(dc)
    odc_s = (odc - np.mean(odc)) / np.std(odc)
    pts_s = (pts - np.mean(pts)) / np.std(pts)
    lista_predittori=[g_s, ga_s, ppda_s, oppda_s, dc_s, odc_s]

    predittori_selezionati=[]
    for index in index_feat_sel:
        predittori_selezionati.append(lista_predittori[index])

    X = np.concatenate((predittori_selezionati), axis=1)

    model = sm.OLS(pts_s, X).fit()
    print(model.summary())

    adjusted_Rsq=np.append(adjusted_Rsq,model.rsquared_adj)


    influence = model.get_influence()
    studentized_residuals = influence.resid_studentized_external
    #print(studentized_residuals)
    index_outliers = [i for i in range(len(studentized_residuals)) if
                      studentized_residuals[i] >= 3 or studentized_residuals[i]<= -3]
    nobs_per_iteration=np.append(nobs_per_iteration,model.nobs)
"""







