from understat import Understat
import json
import aiohttp
import asyncio
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

######SERIE A 2019#######
async def main():
 async with aiohttp.ClientSession() as session:
  understat = Understat(session)
  team = await understat.get_league_table(
    "Serie A", "2019")
  global lista
  lista=json.dumps(team)
  #print(json.dumps(team))
  #print(type(team))
  #print(lista[0])
 ''' for i in range(1, 21):
      xgoal_for_team = lista[i][8]
      name_team = lista[i][0]
      list_xgol_for_team = {name_team: xgoal_for_team}
  print(list_xgol_for_team)'''
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
stats_for_team=[]
'''with open("serieA2019.txt",encoding="utf-8") as f:
    line=f.readline()
    while line:
        line=f.readline()
        line=line.replace("\n","")
        stats_for_team.append(list(line.split(",")))
        print(line)
print(stats_for_team)
for i in range(0, 21):
      xgoal_for_team = stats_for_team[i][8]
      name_team = stats_for_team[i][0]
      list_xgol_for_team = {name_team: xgoal_for_team}
dict_xgol_for_team={stats_for_team[i][0]:float(stats_for_team[i][8]) for i in range(0,21)}'''
#print(dict_xgol_for_team)
df=pd.read_csv("serieA2019.txt")
print(df)
data={"W":df.iloc[:,2],
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
      }
dataframe=pd.DataFrame(data,columns=["W","D","L","G", "GA", "PTS", "xG", "NPxG", "xGA", "NPxGA", "NPxGD", "PPDA", "OPPDA", "DC", "ODC", "xPTS"])
corrMatrix=dataframe.corr()
print(corrMatrix)
sn.heatmap(corrMatrix,annot=True)
plt.show()
print(df.iloc[:,0])


