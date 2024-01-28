import matplotlib.pyplot as plt
import accessodati as acv
import seaborn as sns
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression

predictors=["G","GA","PPDA","OPPDA","DC","ODC"]
'''
#regressione lineare: PTS target
#G,GA,PPDA,OPPDA,DC,ODC  predictors
predictors=["G","GA","PPDA","OPPDA","DC","ODC"]
for i in range(len(predictors)):
    plt.scatter(regressioni.pts_s,regressioni.X[:,i])
    plt.title("PTS vs "+predictors[i])
    plt.ylabel("PTS")
    plt.xlabel(predictors[i])
    plt.show()
'''

'''
#RESIDUI
predictors=["G","GA","PPDA","OPPDA","DC","ODC"]
for i in range(len(predictors)):
    sns.residplot(x=regressioni.X[:,0],y=regressioni.pts_s)
    plt.title("PTS vs"+predictors[i])
    plt.show()
    '''


#plot con ciascun predittore alla volta
'''
for i in range(len(predictors)):
    x=regressioni.X[:,i].reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(x, regressioni.pts_s, test_size=0.2, random_state=42)
    model=LinearRegression()
    visualizer=ResidualsPlot(model,title="PTS vs "+predictors[i])
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()


    visualizer = ResidualsPlot(model, hist=False, qqplot=True,title="PTS vs "+predictors[i])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    '''

#se residui non mostrano patter definito, ma sono randomicamente dispersi attorno allo zero, modello lineare fitta bene
#valuto dal grafico se esiste eterochedasticità o no per capire se varianza degli errori è non-costante oppure costante
X_train, X_test, y_train, y_test = train_test_split(acv.X_six_std, acv.pts_s, test_size=0.2, random_state=42)
model=LinearRegression()
visualizer=ResidualsPlot(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()



#trovare outliers tramite residui studentizzati (valore maggiore di 3 indica outlier) per migliorare il fit lineare
import statsmodels.api as sm
from statsmodels.formula.api import ols







