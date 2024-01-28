import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import accessodati as acv
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

##PCA ANALYSIS##
"""
exp_var_list=np.array([])
X=acv.X_SUPREMO
n_features=int(X.shape[0])
#print(n_features)
for i in range(1,n_features):
    pca=PCA(n_components=i)
    principal_components=pca.fit_transform(X)
    exp_var_list=np.append(exp_var_list,pca.explained_variance_ratio_[i-1])
    #print(exp_var_list)
    cumsumus=np.cumsum(exp_var_list)

plt.plot(np.arange(1,n_features),exp_var_list,label="Explained Variance Ratio",color="darksalmon",linestyle="--")
plt.plot(np.arange(1,n_features),cumsumus,label="Cumulative Explained Variance Ratio",color="firebrick")
plt.legend()
plt.xlabel("Number of principal components")
plt.ylabel("Explained Variance Ratio")
plt.show()



from sklearn.metrics import mean_squared_error as mse
#RISULTATI OTTENUTI DAL CODICE PRECEDENTE
#30:140 principal components for X_SUPREMO
rmse_PCR=[]
r2scores=[]
X=acv.X_SUPREMO
#number principal components
for i in range(20,80):
    pca=PCA(n_components=i)
    X_new=pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_new, acv.pts_s, test_size=0.2, random_state=201)

    regr = LinearRegression()
    regr.fit(X_train,y_train)

    pred=regr.predict(X_test)
    rmse_PCR.append(np.sqrt(mse(y_test,pred)))
    r2scores.append(regr.score(X_test,y_test))
    print(i,regr.score(X_test,y_test))
fig,axs=plt.subplots(2)
axs[0].plot(range(20,80),rmse_PCR,label="RMSE",color="black")
axs[1].plot(range(20,80),r2scores,label="R2",color="firebrick")
axs[1].set_xlabel("Number of principal components")
#axs[0].set_xlabel("Number principal components")
axs[1].set_ylabel("R2")
axs[0].set_ylabel("RMSE")
plt.show()
"""


#####LASSO AND RIDGE REGRESSION#####

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
import random
#X=acv.X_SUPREMO
X= np.concatenate((acv.X_four_std,acv.X_more_std),axis=1)
#PRE-PROCESSING
X=np.delete(X,[139, 5 ,12 , 13 ,119, 8, 104, 123, 30, 22, 56, 110, 131, 32, 113, 28, 35, 43, 47, 51, 36, 46, 39,
                     42, 60, 97, 98, 99, 101, 106, 147] ,axis=1)

y=acv.pts_s
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###############
####LASSO#####
#############

"""
alpha_to_tune=np.arange(0.05,1,0.05)
r2_scores=np.array([])
for i in alpha_to_tune:
    rg = Lasso(alpha=i)
    rg.fit(X_train, y_train)
    r2_scores=np.append(r2_scores,rg.score(X_test,y_test))
plt.plot(alpha_to_tune,r2_scores)
plt.title("Lasso hyperparameter tuning")
plt.xlabel("Lambda")
plt.ylabel("R2_test")
plt.show()
"""

#LASSO CROSS VALIDATION

lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{"alpha": alphas}]
n_folds = 5
k_fold = KFold(5,shuffle=True, random_state=0)
clf = GridSearchCV(lasso, tuned_parameters, cv=k_fold, refit=False)
clf.fit(X, y)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]
print(clf.best_score_)
print(clf.best_index_)
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, "b--")
plt.semilogx(alphas, scores - std_error, "b--")

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel("CV score +/- std error")
plt.xlabel("Lambda")
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.axvline(alphas[clf.best_index_],linestyle="--",color=".5")
plt.plot(alphas[clf.best_index_],np.max(scores),marker="o",color="red",markersize=5)
plt.xlim([alphas[0], alphas[-1]])
plt.title("Lasso hyperparameter tuning")
plt.show()
lasso_cv = LassoCV(alphas=alphas)
k_fold = KFold(5,shuffle=True)
for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print(
        "[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(
            k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])
        )
    )
#[fold 0] alpha: 0.00853, score: 0.98870


#LASSO OPTIMAL

rg=Lasso(alpha=alphas[clf.best_index_])
k_fold=KFold(5,shuffle=True)
scorecv = cross_val_score(rg, X, y, cv=k_fold)
#print(scorecv)
#print(np.mean(scorecv))
rg.fit(X, y)
#print("Lasso score:")
#print(rg.score(X_test,y_test))
lista_coeff=rg.coef_
#print("Coeff:")
#print(lista_coeff)
#RETURN
"""
Lasso score:
0.9867573329603275
Coeff:
[ 0.1790234  -0.09521055  0.          0.01301915 -0.         -0.
  0.          0.         -0.         -0.          0.00585999  0.
  0.         -0.          0.          0.          0.          0.
 -0.          0.          0.00184801 -0.         -0.         -0.00426435
 -0.         -0.         -0.         -0.         -0.         -0.01353228
 -0.00355049 -0.         -0.         -0.          0.         -0.
  0.          0.0496361  -0.         -0.00362233  0.01032051  0.
 -0.         -0.          0.46301413  0.         -0.          0.08317794
  0.          0.         -0.         -0.         -0.          0.
 -0.         -0.         -0.203332   -0.18997551  0.         -0.
 -0.          0.         -0.         -0.          0.          0.#
  0.          0.         -0.          0.          0.          0.
 -0.          0.          0.          0.          0.00208002  0.
  0.013213   -0.         -0.          0.         -0.         -0.
 -0.         -0.0018886  -0.01065603 -0.          0.         -0.
 -0.         -0.00675252 -0.00132899 -0.01821671 -0.          0.
 -0.01804341 -0.         -0.         -0.          0.          0.
  0.         -0.          0.          0.          0.          0.
  0.         -0.         -0.         -0.         -0.         -0.00753145
 -0.01008735 -0.         -0.         -0.          0.          0.
  0.         -0.          0.          0.          0.         -0.
  0.          0.         -0.         -0.         -0.         -0.00164818
 -0.          0.         -0.01013463 -0.00230639  0.         -0.
 -0.         -0.         -0.          0.         -0.          0.
  0.          0.          0.         -0.          0.         -0.
  0.          0.          0.         -0.         -0.         -0.
 -0.        ]
"""

dizionario={}
lista_indici_nonnulli=[]
for i in range(0,124):
    if lista_coeff[i]!= 0:
          dizionario[acv.COLONNE_NUOVE[i]]="{0:.5f}".format(lista_coeff[i])
          lista_indici=np.append(lista_indici_nonnulli,i)
print(lista_indici_nonnulli)
print("Number of features that have not null coefficients:")
print(len(dizionario))
print("Feature:coefficient")
print(dizionario)


#RETURN

"""
lista_indici_nonnulli=[  0.   1.   3.  10.  20.  23.  29.  30.  37.  39.  40.  44.  47.  56.
  57.  76.  78.  85.  86.  91.  92.  93.  96. 113. 114. 131. 134. 135.]
Features that have not null coefficients: 28
Feature:coefficient
{'g': '0.17902', 'ga': '-0.09521', 'oppda': '0.01302', 'penalty_shots': '0.00586', 'penalty_xG': '0.00185',
'freekick_shots_against': '-0.00426', 'setpiece_goals_against': '-0.01353', 'penalty_goals_against': '-0.00355',
'vantaggio1_time': '0.04964', 'svantaggio1_time': '-0.00362', 'diff0_shots': '0.01032', 'diff0_goals': '0.46301',
'svantaggio1_goals': '0.08318', 'diff0_goals_against': '-0.20333', 'vantaggio1_goals_against': '-0.18998', 
'timing_1_15_xG': '0.00208', 'timing_31_45_xG': '0.01321', 'timing_46_50_shots_against': '-0.00189', 
'timing_61_75_shots_against': '-0.01066', 'timing_46_50_goals_against': '-0.00675', 'timing_61_75_goals_against': '-0.00133',
'timing_76_plus_goals_against': '-0.01822', 'timing_31_45_xG_against': '-0.01804', 'shot_zones_penalty_area_goals_against': '-0.00753',
'shot_zones_six_yard_box_goals_against': '-0.01009', 'att_standard_shots_against': '-0.00165',
'att_normal_goals_against': '-0.01013', 'att_standard_goals_against': '-0.00231'}
"""



##############
#####RIDGE####
##############

"""
alpha_to_tune=np.arange(0.05,4,0.05)
r2_scores=np.array([])
ridge_coefs=[]
for i in alpha_to_tune:
    rg = Ridge(alpha=i)
    rg.fit(X_train, y_train)
    r2_scores=np.append(r2_scores,rg.score(X_test,y_test))
    ridge_coefs.append(rg.coef_)
print(r2_scores)
plt.plot(alpha_to_tune,r2_scores)
plt.title("Ridge hyperparameter tuning")
plt.xlabel("Lambda")
plt.ylabel("R2_test")
plt.show()
"""

#RIDGE CROSS VALIDATION
"""
ridge = Ridge(random_state=0,max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{"alpha": alphas}]
n_folds = 5
k_fold = KFold(5,shuffle=True)
clf = GridSearchCV(ridge, tuned_parameters, cv=k_fold, refit=False)
clf.fit(X, y)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, "b--")
plt.semilogx(alphas, scores - std_error, "b--")

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel("CV score +/- std error")
plt.xlabel("Lambda")
plt.yticks(list(np.arange(-1,1.2,0.20)))
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.xlim([alphas[0], alphas[-1]])
plt.title("Ridge hyperparameter tuning")
plt.show()
ridge_cv = RidgeCV(alphas=alphas)
k_fold = KFold(5,shuffle=True)
for k, (train, test) in enumerate(k_fold.split(X, y)):
    ridge_cv.fit(X[train], y[train])
    print(
        "[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(
            k, ridge_cv.alpha_, ridge_cv.score(X[test], y[test])
        )
    )
"""
#RETURNS
#[fold 0] alpha: 0.31623, score: 0.92403
#[fold 1] alpha: 0.31623, score: 0.89125
#[fold 2] alpha: 0.31623, score: 0.92617

#RETURNS WITH SHUFFLE
#[fold 0] alpha: 0.31623, score: 0.94767
#[fold 1] alpha: 0.31623, score: 0.86963
#[fold 2] alpha: 0.31623, score: 0.93449

#RIDGE CON LAMBDA OTTIMALE
"""
#rg_vero=Ridge(alpha=0.31623)
#k_fold=KFold(5,shuffle=True)
#scorecv = cross_val_score(rg_vero, X, y, cv=k_fold)
#print(scorecv)
#print(np.mean(scorecv))
#rg_vero.fit(X_train,y_train)
#print("ECCOLOOOO")
#print(rg.score(X_test,y_test))
#0.9446024732712267
#print(rg.coef_)


n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    #ridge.coef_.reshape(-1,157)
    #print(ridge.coef_[0])

    coefs.append(ridge.coef_[0])
    #print(coefs)

    #print(np.shape(ridge.coef_))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("Lambda")
plt.ylabel("Coefficients")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()
"""



