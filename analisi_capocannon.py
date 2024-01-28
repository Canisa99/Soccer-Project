import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
from dateutil.parser import parse
from statsmodels.tsa.filters import bk_filter, hp_filter
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import LinearRegression


dati_capi=pd.read_csv("Capocannonieri.txt")
goals=dati_capi.iloc[:,5].values
n_squads=dati_capi.iloc[:,2].values
goals=goals/n_squads
goals=goals[::-1]
stagioni=dati_capi.iloc[:,0].values
stagioni=stagioni[::-1]
media=np.array([])

###PLOT DELLA SERIE###
print(goals)
fig,ax1 = plt.subplots()
ax1.plot(stagioni,goals,linestyle= "-",color="royalblue",label="Data observed",marker="o",markersize=2.39,linewidth=0.43)
for c in range(0,np.shape(dati_capi)[0]):
      media=np.append(media,np.mean(goals))
ax1.plot(stagioni,media,linestyle= "--",color="royalblue",label="Mean observed",linewidth=0.79)
ax1.set_xlabel("Seasons")
ax1.set_ylabel("Top Scorer Goals per Game")
ax1.set_title("Time Series for Top Scorers")
plt.xticks(rotation=80)
#plt.show()

"""
df=pd.read_csv("nuovo_capocannonieri.txt",parse_dates=['Seasons'])
# Multiplicative Decomposition
result_mul = seasonal_decompose(df["Goals per Game"],period=16, model='multiplicative')

# Additive Decomposition
result_add = seasonal_decompose(df["Goals per Game"],period=16, model='additive')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
"""

#Subtract the line of best fit from the time series.
#The line of best fit may be obtained from a linear regression model with the time steps as the predictor.
#For more complex trends, you may want to use quadratic terms (x^2) in the model.
detrended = signal.detrend(goals,type="linear")
ax1.plot(stagioni,goals-detrended,color="firebrick",linewidth=0.69,label="Trend")
plt.legend()
# HP FILTER
#cycle, trend = hp_filter.hpfilter(goals, 6.25*(1600/4**4))

###PLOT DETRENDED###
"""
ax1.plot(trend,label="Trend")
plt.legend()
fig,ax2 = plt.subplots()
ax2.plot(stagioni,cycle) #cycle sarebbe data-trend
ax2.set_xlabel("Seasons")
ax2.set_ylabel("Top Scorer Goals per Game")
ax2.set_title("Cycle Component")
plt.xticks(rotation=80)
plt.show()
"""

#plot_acf(goals, lags = np.shape(goals)[0]-1)
#print('\n ---- PARTIAL ----')
#plot_pacf(goals,  lags =15)
#plt.show()

result = adfuller(goals, autolag='AIC', regression="ct")
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

def dfullercoeff(vec):
  X = np.array(vec[:len(vec)-1])
  X = X.reshape(-1,1)
  y = vec[1:]
  model = LinearRegression()
  model.fit(X, y)
  print('slope:', model.coef_)
  print('intercept:', model.intercept_)

dfullercoeff(goals)
###WHITE TEST###
"""
from statsmodels.stats.diagnostic import het_white
x=np.arange(1988,2021)
x= sm.add_constant(x)

model = sm.OLS(goals, x).fit()
print(model.summary())

white_test = het_white(model.resid, model.model.exog)

#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

#print results of White's test
print(dict(zip(labels, white_test)))
"""