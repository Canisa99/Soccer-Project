import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Se ho un dataframe lo converto in pickle con dataframe.to_pickle("nome del file pkl")
#df.to_pickle('gca_pass_shot.pkl')
#Una volta salvato il file pickle su pycharm me lo posso leggere con panda
restofdata=pd.read_pickle("gca_pass_shot.pkl")
possession=pd.read_csv("POSSESSION.txt")
def_action=pd.read_csv("DEF_ACTION.txt")
pass_types=pd.read_csv("PASS_TYPES.txt")
punti_frame=pd.read_csv("pts_seriea_2022-2018.txt")
punti=np.array(punti_frame['Pts'])

#possession: "Succ%" "Prog.1" "Squad"
dribl_succ_perc=possession['Succ%']
progr_pass=possession['Prog.1']
squad=possession['Squad']

#def_action: "Succ" "Att 3rd"
press_succ_5s=def_action['Succ']
press_on_opponent_attack=def_action['Att 3rd.1']

#pass_types:
pass_press=pass_types['Press']
cmp=pass_types['Cmp']

data={"Squad":squad,"Succ%":dribl_succ_perc,"Prog":progr_pass,"Press_succ":press_succ_5s,"Press_attack":press_on_opponent_attack,"Pass_press":pass_press,
      "Passes completed":cmp}
dataframe=pd.DataFrame(data=data,columns=["Squad","Succ%","Prog","Press_succ","Press_attack","Pass_press","Passes completed"])
restofdata=restofdata.drop(columns=["Squadra"])
frames=[dataframe,restofdata]
dataframecomplete=pd.concat(frames,axis=1)
#dataframecomplete['Punti']=punti #attenzione vanno in ordine di classifica

#Aggiungo colonna stagioni
anno=np.array([])
for i in range(0,100):
      if i<20: anno=np.append(anno,"2021/2022")
      if 19<i<40: anno=np.append(anno,"2020/2021")
      if 39<i<60: anno=np.append(anno,"2019/2020")
      if 59<i<80: anno=np.append(anno,"2018/2019")
      if 79<i<100: anno=np.append(anno,"2017/2018")
dataframecomplete['Stagione']=anno #è l'ultima colonna

features=["Succ%","Prog","Press_succ","Press_attack","Pass_press","Passes completed","SCA","Def.","Dist. Prog.","xA","Prog.","Dist."]
seasons=["2017/2018","2018/2019","2019/2020","2020/2021"]
squadre=["Napoli","Atalanta","Milan","Roma","Lazio","Juventus","Inter"]
dataframe5=dataframecomplete.iloc[0:20,:] #stagione 2021/2022
dataframe4=dataframecomplete.iloc[20:40,:] #stagione 2020/2021
dataframe3=dataframecomplete.iloc[40:60,:] #stagione 2019/2020
dataframe2=dataframecomplete.iloc[60:80,:] #stagione 2018/2019
dataframe1= dataframecomplete.iloc[80:100,:] #stagione 2017/2018
lista_dataframe=[dataframe1,dataframe2,dataframe3,dataframe4]
for i in range(0,12):
      fig,ax = plt.subplots()
      lista1 = np.array([])
      lista2 = np.array([])
      lista3 = np.array([])
      lista4 = np.array([])
      lista5 = np.array([])
      lista6 = np.array([])
      lista7 = np.array([])
      medie=np.array([])
      for j in lista_dataframe:
            for k in range(0,20):
                  if j.iloc[k,0]=="Napoli":
                        lista1=np.append(lista1,j.iloc[k, i + 1])
                  if j.iloc[k,0]=="Atalanta":
                        lista2=np.append(lista2,j.iloc[k, i + 1])
                  if j.iloc[k,0]=="Milan":
                        lista3=np.append(lista3,j.iloc[k, i + 1])
                  if j.iloc[k,0]=="Inter":
                        lista4=np.append(lista4,j.iloc[k, i + 1])
                  if j.iloc[k,0]=="Juventus":
                        lista5=np.append(lista5,j.iloc[k, i + 1])
                  if j.iloc[k,0]=="Lazio":
                        lista6=np.append(lista6,j.iloc[k, i + 1])
                  if j.iloc[k,0]=="Roma":
                        lista7=np.append(lista7,j.iloc[k, i + 1])
      ax.plot(seasons,lista1,linestyle= "-", label="Napoli",color="red")
      ax.plot(seasons, lista2, linestyle="-", label="Atalanta",color="blue")
      ax.plot(seasons, lista3, linestyle="-", label="Milan", color="yellow")
      ax.plot(seasons, lista4, linestyle="-", label="Inter", color="orange")
      ax.plot(seasons, lista5, linestyle="-", label="Juventus", color="green")
      ax.plot(seasons, lista6, linestyle="-", label="Lazio", color="lightskyblue")
      ax.plot(seasons, lista7, linestyle="-", label="Roma", color="fuchsia")
      #ANCHE LE MEDIE
      """
      lista_liste = [lista1, lista2, lista3, lista4, lista5, lista6, lista7]
      for z in range(0,7):
            for t in range(0, 4):
                  medie = np.append(medie, np.mean(lista_liste[z]))
      medie=np.reshape(medie,(7,4))
      ax.plot(seasons, medie[0,:], linestyle="--", color="red")
      ax.plot(seasons, medie[1,:], linestyle="--", color="blue")
      ax.plot(seasons, medie[2,:], linestyle="--", color="yellow")
      ax.plot(seasons, medie[3,:], linestyle="--", color="orange")
      ax.plot(seasons, medie[4,:], linestyle="--", color="green")
      ax.plot(seasons, medie[5,:], linestyle="--", color="lightskyblue")
      ax.plot(seasons, medie[6,:], linestyle="--", color="fuchsia")
      """
      ax.legend(fontsize="x-small")
      ax.set_xlabel("Seasons")
      ax.set_ylabel(features[i])
      ax.set_title("Time Series Analysis for " + features[i])
      plt.show()








