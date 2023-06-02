import pandas as pd
import numpy as np
import json
import functionz
from sklearn.preprocessing import StandardScaler

#CARICO DATASET SERIE A DI OGNI ANNO
df_2020=pd.read_csv("Data/seriea2020.txt")
df_2019=pd.read_csv("Data/serieA2019.txt")
df_2018=pd.read_csv("Data/seriea2018.txt")
df_2017=pd.read_csv("Data/seriea2017.txt")
df_2016=pd.read_csv("Data/seriea2016.txt")
df_2015=pd.read_csv("Data/seriea2015.txt")
df_2014=pd.read_csv("Data/seriea2014.txt")

#CREO UN DATAFRAME 140Xnfeatures DOVE UNISCO TUTTI I DATAFRAME DI OGNI ANNO
lista_dataframe_complete=[df_2020,df_2019,df_2018,df_2017,df_2016,df_2015,df_2014]
dataframe_definitivo=pd.concat(lista_dataframe_complete,axis=0,ignore_index=True)


#CARICO FILE JSON (Statistiche per ogni singola squadra per ogni anno, più dettagliate come possesso palla eccetera)
with open("Data/STOCA19.json") as file:
    data2019=json.load(file)

with open("Data/STOCA19.json",'w') as file:
    json.dump(data2019,file)

with open("Data/STOCA20.json") as file:
    data2020=json.load(file)

with open("Data/STOCA20.json",'w') as file:
    json.dump(data2020,file)

with open("Data/STOCA18.json") as file:
    data2018=json.load(file)

with open("Data/STOCA18.json",'w') as file:
    json.dump(data2018,file)

with open("Data/STOCA17.json") as file:
    data2017=json.load(file)

with open("Data/STOCA17.json",'w') as file:
    json.dump(data2017,file)

with open("Data/STOCA16.json") as file:
    data2016=json.load(file)

with open("Data/STOCA16.json",'w') as file:
    json.dump(data2016,file)

with open("Data/STOCA15.json") as file:
    data2015=json.load(file)

with open("Data/STOCA15.json",'w') as file:
    json.dump(data2015,file)

with open("Data/STOCA14.json") as file:
    data2014=json.load(file)

with open("Data/STOCA14.json",'w') as file:
    json.dump(data2014,file)
#CONCATENO LE LISTE
lista_stats_complete=data2020+data2019+data2018+data2017+data2016+data2015+data2014

#COME ACCEDO A QUESTO FILE?
"""
PRIME QUADRE METTO L'INDICE CHE MI INDICA LA SQUADRA, POI LE ALTRE QUADRE ACCEDO AI VARI CAMPI
DEL DIZIONARIO TIPO "SITUATION" "OPENPLAY" "SHOTS" OPPURE "SITUATION" "OPENPLAY" "AGAINST"....
ESEMPIO:
print(lista_stats_complete[0]['situation']['OpenPlay'])
RESTITUISCE {'shots': 413, 'goals': 68, 'xG': 62.1393936611712, 'against': {'shots': 293, 'goals': 24, 'xG': 29.055169105064124}}
"""

#GET SITUATION STATS
#OPENPLAY
openplay_shots=functionz.get_stats(lista_stats_complete,"situation","OpenPlay","shots").reshape(-1,1)
openplay_goals=functionz.get_stats(lista_stats_complete,"situation","OpenPlay","goals").reshape(-1,1)
openplay_xG=functionz.get_stats(lista_stats_complete,"situation","OpenPlay","xG").reshape(-1,1)
openplay_shots_against=functionz.get_stats_against(lista_stats_complete,"situation","OpenPlay","shots").reshape(-1,1)
openplay_goals_against=functionz.get_stats_against(lista_stats_complete,"situation","OpenPlay","goals").reshape(-1,1)
openplay_xG_against=functionz.get_stats_against(lista_stats_complete,"situation","OpenPlay","xG").reshape(-1,1)
#FROMCORNER
fromcorner_shots=functionz.get_stats(lista_stats_complete,"situation","FromCorner","shots").reshape(-1,1)
fromcorner_goals=functionz.get_stats(lista_stats_complete,"situation","FromCorner","goals").reshape(-1,1)
fromcorner_xG=functionz.get_stats(lista_stats_complete,"situation","FromCorner","shots").reshape(-1,1)
fromcorner_shots_against=functionz.get_stats_against(lista_stats_complete,"situation","FromCorner","shots").reshape(-1,1)
fromcorner_goals_against=functionz.get_stats_against(lista_stats_complete,"situation","FromCorner","goals").reshape(-1,1)
fromcorner_xG_against=functionz.get_stats_against(lista_stats_complete,"situation","FromCorner","xG").reshape(-1,1)
#FREEKICK
freekick_shots=functionz.get_stats(lista_stats_complete,"situation","DirectFreekick","shots").reshape(-1,1)
freekick_goals=functionz.get_stats(lista_stats_complete,"situation","DirectFreekick","goals").reshape(-1,1)
freekick_xG=functionz.get_stats(lista_stats_complete,"situation","DirectFreekick","xG").reshape(-1,1)
freekick_shots_against=functionz.get_stats_against(lista_stats_complete,"situation","DirectFreekick","shots").reshape(-1,1)
freekick_goals_against=functionz.get_stats_against(lista_stats_complete,"situation","DirectFreekick","goals").reshape(-1,1)
freekick_xG_against=functionz.get_stats_against(lista_stats_complete,"situation","DirectFreekick","xG").reshape(-1,1)
#SETPIECE
setpiece_shots=functionz.get_stats(lista_stats_complete,"situation","SetPiece","shots").reshape(-1,1)
setpiece_goals=functionz.get_stats(lista_stats_complete,"situation","SetPiece","goals").reshape(-1,1)
setpiece_xG=functionz.get_stats(lista_stats_complete,"situation","SetPiece","xG").reshape(-1,1)
setpiece_shots_against=functionz.get_stats_against(lista_stats_complete,"situation","SetPiece","shots").reshape(-1,1)
setpiece_goals_against=functionz.get_stats_against(lista_stats_complete,"situation","SetPiece","goals").reshape(-1,1)
setpiece_xG_against=functionz.get_stats_against(lista_stats_complete,"situation","SetPiece","xG").reshape(-1,1)
#PENALTY
penalty_shots=functionz.get_stats(lista_stats_complete,"situation","Penalty","shots").reshape(-1,1)
penalty_goals=functionz.get_stats(lista_stats_complete,"situation","Penalty","goals").reshape(-1,1)
penalty_xG=functionz.get_stats(lista_stats_complete,"situation","Penalty","xG").reshape(-1,1)
penalty_shots_against=functionz.get_stats_against(lista_stats_complete,"situation","Penalty","shots").reshape(-1,1)
penalty_goals_against=functionz.get_stats_against(lista_stats_complete,"situation","Penalty","goals").reshape(-1,1)
penalty_xG_against=functionz.get_stats_against(lista_stats_complete,"situation","Penalty","xG").reshape(-1,1)

#GET GAMESTATE STATS
#GOAL DIFF 0
diff0_time=functionz.get_stats(lista_stats_complete,"gameState","Goal diff 0","time").reshape(-1,1)
diff0_shots=functionz.get_stats(lista_stats_complete,"gameState","Goal diff 0","shots").reshape(-1,1)
diff0_goals=functionz.get_stats(lista_stats_complete,"gameState","Goal diff 0","goals").reshape(-1,1)
diff0_xG=functionz.get_stats(lista_stats_complete,"gameState","Goal diff 0","xG").reshape(-1,1)
diff0_shots_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff 0","shots").reshape(-1,1)
diff0_goals_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff 0","goals").reshape(-1,1)
diff0_xG_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff 0","xG").reshape(-1,1)
#GOAL DIFF +1
vantaggio1_time=functionz.get_stats(lista_stats_complete,"gameState","Goal diff +1","time").reshape(-1,1)
vantaggio1_shots=functionz.get_stats(lista_stats_complete,"gameState","Goal diff +1","shots").reshape(-1,1)
vantaggio1_goals=functionz.get_stats(lista_stats_complete,"gameState","Goal diff +1","goals").reshape(-1,1)
vantaggio1_xG=functionz.get_stats(lista_stats_complete,"gameState","Goal diff +1","xG").reshape(-1,1)
vantaggio1_shots_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff +1","shots").reshape(-1,1)
vantaggio1_goals_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff +1","goals").reshape(-1,1)
vantaggio1_xG_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff +1","xG").reshape(-1,1)
#GOAL DIFF >+1
vantaggio_multiplo_time=functionz.get_stats(lista_stats_complete,"gameState","Goal diff > +1","time").reshape(-1,1)
vantaggio_multiplo_shots=functionz.get_stats(lista_stats_complete,"gameState","Goal diff > +1","shots").reshape(-1,1)
vantaggio_multiplo_goals=functionz.get_stats(lista_stats_complete,"gameState","Goal diff > +1","goals").reshape(-1,1)
vantaggio_multiplo_xG=functionz.get_stats(lista_stats_complete,"gameState","Goal diff > +1","xG").reshape(-1,1)
vantaggio_multiplo_shots_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff > +1","shots").reshape(-1,1)
vantaggio_multiplo_goals_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff > +1","goals").reshape(-1,1)
vantaggio_multiplo_xG_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff > +1","xG").reshape(-1,1)
#GOAL DIFF -1
svantaggio1_time=functionz.get_stats(lista_stats_complete,"gameState","Goal diff -1","time").reshape(-1,1)
svantaggio1_shots=functionz.get_stats(lista_stats_complete,"gameState","Goal diff -1","shots").reshape(-1,1)
svantaggio1_goals=functionz.get_stats(lista_stats_complete,"gameState","Goal diff -1","goals").reshape(-1,1)
svantaggio1_xG=functionz.get_stats(lista_stats_complete,"gameState","Goal diff -1","xG").reshape(-1,1)
svantaggio1_shots_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff -1","shots").reshape(-1,1)
svantaggio1_goals_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff -1","goals").reshape(-1,1)
svantaggio1_xG_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff -1","xG").reshape(-1,1)
#GOAL DIFF < -1 HA UN PROBLEMA, IL DIZIONARIO è CREATO SOLO PER 119 SQUADRE SU 140 QUINDI NOGOOD.
"""
#GOAL DIFF <-1
svantaggio_multiplo_time=functionz.get_stats(lista_stats_complete,"gameState","Goal diff < -1","time").reshape(-1,1)
svantaggio_multiplo_shots=functionz.get_stats(lista_stats_complete,"gameState","Goal diff < -1","shots").reshape(-1,1)
svantaggio_multiplo_goals=functionz.get_stats(lista_stats_complete,"gameState","Goal diff < -1","goals").reshape(-1,1)
svantaggio_multiplo_xG=functionz.get_stats(lista_stats_complete,"gameState","Goal diff < -1","xG").reshape(-1,1)
svantaggio_multiplo_shots_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff < -1","shots").reshape(-1,1)
svantaggio_multiplo_goals_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff < -1","goals").reshape(-1,1)
svantaggio_multiplo_xG_against=functionz.get_stats_against(lista_stats_complete,"gameState","Goal diff < -1","xG").reshape(-1,1)
"""

#GET TIMING STATS
timing_1_15_shots = functionz.get_stats(lista_stats_complete, 'timing', '1-15','shots').reshape(-1,1)
timing_16_30_shots = functionz.get_stats(lista_stats_complete, 'timing', '16-30','shots').reshape(-1,1)
timing_31_45_shots = functionz.get_stats(lista_stats_complete, 'timing', '31-45','shots').reshape(-1,1)
timing_46_50_shots = functionz.get_stats(lista_stats_complete, 'timing', '46-60','shots').reshape(-1,1)
timing_61_75_shots = functionz.get_stats(lista_stats_complete, 'timing', '61-75','shots').reshape(-1,1)
timing_76_plus_shots = functionz.get_stats(lista_stats_complete, 'timing', '76+','shots').reshape(-1,1)

timing_1_15_goals = functionz.get_stats(lista_stats_complete, 'timing', '1-15','goals').reshape(-1,1)
timing_16_30_goals = functionz.get_stats(lista_stats_complete, 'timing', '16-30','goals').reshape(-1,1)
timing_31_45_goals = functionz.get_stats(lista_stats_complete, 'timing', '31-45','goals').reshape(-1,1)
timing_46_50_goals = functionz.get_stats(lista_stats_complete, 'timing', '46-60','goals').reshape(-1,1)
timing_61_75_goals = functionz.get_stats(lista_stats_complete, 'timing', '61-75','goals').reshape(-1,1)
timing_76_plus_goals = functionz.get_stats(lista_stats_complete, 'timing', '76+','goals').reshape(-1,1)

timing_1_15_xG = functionz.get_stats(lista_stats_complete, 'timing', '1-15','xG').reshape(-1,1)
timing_16_30_xG = functionz.get_stats(lista_stats_complete, 'timing', '16-30','xG').reshape(-1,1)
timing_31_45_xG = functionz.get_stats(lista_stats_complete, 'timing', '31-45','xG').reshape(-1,1)
timing_46_50_xG = functionz.get_stats(lista_stats_complete, 'timing', '46-60','xG').reshape(-1,1)
timing_61_75_xG = functionz.get_stats(lista_stats_complete, 'timing', '61-75','xG').reshape(-1,1)
timing_76_plus_xG = functionz.get_stats(lista_stats_complete, 'timing', '76+','xG').reshape(-1,1)

timing_1_15_shots_against = functionz.get_stats_against(lista_stats_complete, 'timing', '1-15','shots').reshape(-1,1)
timing_16_30_shots_against = functionz.get_stats_against(lista_stats_complete, 'timing', '16-30','shots').reshape(-1,1)
timing_31_45_shots_against = functionz.get_stats_against(lista_stats_complete, 'timing', '31-45','shots').reshape(-1,1)
timing_46_50_shots_against = functionz.get_stats_against(lista_stats_complete, 'timing', '46-60','shots').reshape(-1,1)
timing_61_75_shots_against = functionz.get_stats_against(lista_stats_complete, 'timing', '61-75','shots').reshape(-1,1)
timing_76_plus_shots_against = functionz.get_stats_against(lista_stats_complete, 'timing', '76+','shots').reshape(-1,1)

timing_1_15_goals_against = functionz.get_stats_against(lista_stats_complete, 'timing', '1-15','goals').reshape(-1,1)
timing_16_30_goals_against = functionz.get_stats_against(lista_stats_complete, 'timing', '16-30','goals').reshape(-1,1)
timing_31_45_goals_against = functionz.get_stats_against(lista_stats_complete, 'timing', '31-45','goals').reshape(-1,1)
timing_46_50_goals_against = functionz.get_stats_against(lista_stats_complete, 'timing', '46-60','goals').reshape(-1,1)
timing_61_75_goals_against = functionz.get_stats_against(lista_stats_complete, 'timing', '61-75','goals').reshape(-1,1)
timing_76_plus_goals_against = functionz.get_stats_against(lista_stats_complete, 'timing', '76+','goals').reshape(-1,1)

timing_1_15_xG_against = functionz.get_stats_against(lista_stats_complete, 'timing', '1-15','xG').reshape(-1,1)
timing_16_30_xG_against = functionz.get_stats_against(lista_stats_complete, 'timing', '16-30','xG').reshape(-1,1)
timing_31_45_xG_against = functionz.get_stats_against(lista_stats_complete, 'timing', '31-45','xG').reshape(-1,1)
timing_46_50_xG_against = functionz.get_stats_against(lista_stats_complete, 'timing', '46-60','xG').reshape(-1,1)
timing_61_75_xG_against = functionz.get_stats_against(lista_stats_complete, 'timing', '61-75','xG').reshape(-1,1)
timing_76_plus_xG_against = functionz.get_stats_against(lista_stats_complete, 'timing', '76+','xG').reshape(-1,1)

#GET SHOT ZONES STATS
shot_zones_out_of_box__shots = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotOboxTotal','shots').reshape(-1,1)
shot_zones_penalty_area_shots = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotPenaltyArea','shots').reshape(-1,1)
shot_zones_six_yard_box_shots = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotSixYardBox','shots').reshape(-1,1)

shot_zones_out_of_box_goals = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotOboxTotal','goals').reshape(-1,1)
shot_zones_penalty_area_goals = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotPenaltyArea','goals').reshape(-1,1)
shot_zones_six_yard_box_goals = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotSixYardBox','goals').reshape(-1,1)

shot_zones_out_of_box_xG = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotOboxTotal','xG').reshape(-1,1)
shot_zones_penalty_area_xG = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotPenaltyArea','xG').reshape(-1,1)
shot_zones_six_yard_box_xG = functionz.get_stats(lista_stats_complete, 'shotZone', 'shotSixYardBox','xG').reshape(-1,1)

shot_zones_out_of_box_shots_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotOboxTotal','shots').reshape(-1,1)
shot_zones_penalty_area_shots_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotPenaltyArea','shots').reshape(-1,1)
shot_zones_six_yard_box_shots_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotSixYardBox','shots').reshape(-1,1)

shot_zones_out_of_box_goals_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotOboxTotal','goals').reshape(-1,1)
shot_zones_penalty_area_goals_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotPenaltyArea','goals').reshape(-1,1)
shot_zones_six_yard_box_goals_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotSixYardBox','goals').reshape(-1,1)

shot_zones_out_of_box_xG_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotOboxTotal','xG').reshape(-1,1)
shot_zones_penalty_area_xG_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotPenaltyArea','xG').reshape(-1,1)
shot_zones_six_yard_box_xG_against = functionz.get_stats_against(lista_stats_complete, 'shotZone', 'shotSixYardBox','xG').reshape(-1,1)

#GET ATTACK_SPEED STATS
att_normal_shots=functionz.get_stats(lista_stats_complete,"attackSpeed","Normal","shots").reshape(-1,1)
att_standard_shots=functionz.get_stats(lista_stats_complete,"attackSpeed","Standard","shots").reshape(-1,1)
att_slow_shots=functionz.get_stats(lista_stats_complete,"attackSpeed","Slow","shots").reshape(-1,1)
att_fast_shots=functionz.get_stats(lista_stats_complete,"attackSpeed","Fast","shots").reshape(-1,1)

att_normal_goals=functionz.get_stats(lista_stats_complete,"attackSpeed","Normal","goals").reshape(-1,1)
att_standard_goals=functionz.get_stats(lista_stats_complete,"attackSpeed","Standard","goals").reshape(-1,1)
att_slow_goals=functionz.get_stats(lista_stats_complete,"attackSpeed","Slow","goals").reshape(-1,1)
att_fast_goals=functionz.get_stats(lista_stats_complete,"attackSpeed","Fast","goals").reshape(-1,1)

att_normal_xG=functionz.get_stats(lista_stats_complete,"attackSpeed","Normal","xG").reshape(-1,1)
att_standard_xG=functionz.get_stats(lista_stats_complete,"attackSpeed","Standard","xG").reshape(-1,1)
att_slow_xG=functionz.get_stats(lista_stats_complete,"attackSpeed","Slow","xG").reshape(-1,1)
att_fast_xG=functionz.get_stats(lista_stats_complete,"attackSpeed","Fast","xG").reshape(-1,1)

att_normal_shots_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Normal","shots").reshape(-1,1)
att_standard_shots_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Standard","shots").reshape(-1,1)
att_slow_shots_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Slow","shots").reshape(-1,1)
att_fast_shots_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Fast","shots").reshape(-1,1)

att_normal_goals_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Normal","goals").reshape(-1,1)
att_standard_goals_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Standard","goals").reshape(-1,1)
att_slow_goals_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Slow","goals").reshape(-1,1)
att_fast_goals_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Fast","goals").reshape(-1,1)

att_normal_xG_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Normal","xG").reshape(-1,1)
att_standard_xG_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Standard","xG").reshape(-1,1)
att_slow_xG_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Slow","xG").reshape(-1,1)
att_fast_xG_against=functionz.get_stats_against(lista_stats_complete,"attackSpeed","Fast","xG").reshape(-1,1)

#GET RESULT STATS
res_savedshot_shots=functionz.get_stats(lista_stats_complete,"result","SavedShot","shots").reshape(-1,1)
res_missedshot_shots=functionz.get_stats(lista_stats_complete,"result","MissedShots","shots").reshape(-1,1)
res_blockedshot_shots=functionz.get_stats(lista_stats_complete,"result","BlockedShot","shots").reshape(-1,1)
res_goal_shots=functionz.get_stats(lista_stats_complete,"result","Goal","shots").reshape(-1,1)
res_shotonpost_shots=functionz.get_stats(lista_stats_complete,"result","ShotOnPost","shots").reshape(-1,1)

res_savedshot_shots_against=functionz.get_stats_against(lista_stats_complete,"result","SavedShot","shots").reshape(-1,1)
res_missedshot_shots_against=functionz.get_stats_against(lista_stats_complete,"result","MissedShots","shots").reshape(-1,1)
res_blockedshot_shots_against=functionz.get_stats_against(lista_stats_complete,"result","BlockedShot","shots").reshape(-1,1)
res_goal_shots_against=functionz.get_stats_against(lista_stats_complete,"result","Goal","shots").reshape(-1,1)
res_shotonpost_shots_against=functionz.get_stats_against(lista_stats_complete,"result","ShotOnPost","shots").reshape(-1,1)

res_savedshot_xG=functionz.get_stats(lista_stats_complete,"result","SavedShot","xG").reshape(-1,1)
res_missedshot_xG=functionz.get_stats(lista_stats_complete,"result","MissedShots","xG").reshape(-1,1)
res_blockedshot_xG=functionz.get_stats(lista_stats_complete,"result","BlockedShot","xG").reshape(-1,1)
res_goalshot_xG=functionz.get_stats(lista_stats_complete,"result","Goal","xG").reshape(-1,1)
res_shotonpost_xG=functionz.get_stats(lista_stats_complete,"result","ShotOnPost","xG").reshape(-1,1)

res_savedshot_xG_against=functionz.get_stats_against(lista_stats_complete,"result","SavedShot","xG").reshape(-1,1)
res_missedshot_xG_against=functionz.get_stats_against(lista_stats_complete,"result","MissedShots","xG").reshape(-1,1)
res_blockedshot_xG_against=functionz.get_stats_against(lista_stats_complete,"result","BlockedShot","xG").reshape(-1,1)
res_goalshot_xG_against=functionz.get_stats_against(lista_stats_complete,"result","Goal","xG").reshape(-1,1)
res_shotonpost_xG_against=functionz.get_stats_against(lista_stats_complete,"result","ShotOnPost","xG").reshape(-1,1)

#CONCATENIAMO TUTTO

X_situation=np.concatenate((openplay_shots,fromcorner_shots,freekick_shots,setpiece_shots,penalty_shots,openplay_goals,fromcorner_goals,freekick_goals,
                          setpiece_goals,penalty_goals,openplay_xG,fromcorner_xG,freekick_xG,setpiece_xG,penalty_xG,
                          openplay_shots_against,fromcorner_shots_against,freekick_shots_against,setpiece_shots_against,
                          penalty_shots_against,openplay_goals_against,fromcorner_goals_against,freekick_goals_against,
                          setpiece_goals_against,penalty_goals_against,openplay_xG_against,fromcorner_xG_against,freekick_xG_against,
                          setpiece_xG_against,penalty_xG_against),axis=1)

X_gamestate=np.concatenate((diff0_time,vantaggio1_time,vantaggio_multiplo_time,svantaggio1_time,diff0_shots,vantaggio1_shots,vantaggio_multiplo_shots,
                         svantaggio1_shots,diff0_goals,vantaggio1_goals,vantaggio_multiplo_goals,svantaggio1_goals,
                         diff0_xG,vantaggio1_xG,vantaggio_multiplo_xG,svantaggio1_xG,
                         diff0_shots_against,vantaggio1_shots_against,vantaggio_multiplo_shots_against,svantaggio1_shots_against,
                         diff0_goals_against,vantaggio1_goals_against,vantaggio_multiplo_goals_against,svantaggio1_goals_against,
                         diff0_xG_against,vantaggio1_xG_against,vantaggio_multiplo_xG_against,svantaggio1_xG_against),axis=1)

X_timing = np.concatenate(( timing_1_15_shots, timing_16_30_shots, timing_31_45_shots, timing_46_50_shots,
                            timing_61_75_shots, timing_76_plus_shots,
                            timing_1_15_goals, timing_16_30_goals, timing_31_45_goals, timing_46_50_goals,
                            timing_61_75_goals, timing_76_plus_goals,
                            timing_1_15_xG, timing_16_30_xG, timing_31_45_xG, timing_46_50_xG,
                            timing_61_75_xG, timing_76_plus_xG,
                            timing_1_15_shots_against, timing_16_30_shots_against, timing_31_45_shots_against,
                            timing_46_50_shots_against, timing_61_75_shots_against, timing_76_plus_shots_against,
                            timing_1_15_goals_against, timing_16_30_goals_against, timing_31_45_goals_against,
                            timing_46_50_goals_against, timing_61_75_goals_against, timing_76_plus_goals_against,
                            timing_1_15_xG_against, timing_16_30_xG_against, timing_31_45_xG_against,
                            timing_46_50_xG_against, timing_61_75_xG_against, timing_76_plus_xG_against), axis=1)
X_shotZones = np.concatenate((shot_zones_out_of_box__shots,shot_zones_penalty_area_shots,shot_zones_six_yard_box_shots,
                              shot_zones_out_of_box_goals,shot_zones_penalty_area_goals,shot_zones_six_yard_box_goals,
                              shot_zones_out_of_box_xG,shot_zones_penalty_area_xG,shot_zones_six_yard_box_xG,
                              shot_zones_out_of_box_shots_against,shot_zones_penalty_area_shots_against,shot_zones_six_yard_box_shots_against,
                              shot_zones_out_of_box_goals_against,shot_zones_penalty_area_goals_against,shot_zones_six_yard_box_goals_against,
                              shot_zones_out_of_box_xG_against,shot_zones_penalty_area_xG_against,shot_zones_six_yard_box_xG_against),axis = 1)

X_attack=np.concatenate((att_normal_shots,att_standard_shots,att_slow_shots,att_fast_shots,
                          att_normal_goals,att_standard_goals,att_slow_goals,att_fast_goals,
                          att_normal_xG,att_standard_xG,att_slow_xG,att_fast_xG,
                          att_normal_shots_against,att_standard_shots_against,att_slow_shots_against,att_fast_shots_against,
                          att_normal_goals_against,att_standard_goals_against,att_slow_goals_against,att_fast_goals_against,
                          att_normal_xG_against,att_standard_xG_against,att_slow_xG_against,att_fast_xG_against),axis=1)

X_result=np.concatenate((res_savedshot_shots,res_missedshot_shots,res_blockedshot_shots,res_goal_shots,res_shotonpost_shots,
                          res_savedshot_xG,res_missedshot_xG,res_blockedshot_xG,res_goalshot_xG,res_shotonpost_xG,
                          res_savedshot_xG_against,res_missedshot_xG_against,res_blockedshot_xG_against,res_goalshot_xG_against,res_shotonpost_xG_against),axis=1)

#THE SIX
g=functionz.G(dataframe_definitivo).reshape(-1,1) #5
ga=functionz.GA(dataframe_definitivo).reshape(-1,1) #6
ppda=functionz.PPDA(dataframe_definitivo).reshape(-1,1) #13
oppda=functionz.OPPDA(dataframe_definitivo).reshape(-1,1) #14
dc=functionz.DC(dataframe_definitivo).reshape(-1,1) #15
odc=functionz.ODC(dataframe_definitivo).reshape(-1,1) #16
pts=functionz.PTS(dataframe_definitivo).reshape(-1,1) #7
#THE SIX EXPANDED
xg=functionz.xG(dataframe_definitivo).reshape(-1,1) #8
npxg=functionz.NPxG(dataframe_definitivo).reshape(-1,1) #9
xga=functionz.xGA(dataframe_definitivo).reshape(-1,1) #10
npxga=functionz.NPxGA(dataframe_definitivo).reshape(-1,1) #11
npxgd=functionz.NPxGD(dataframe_definitivo).reshape(-1,1) #12

#scaling features
g_s=(g-np.mean(g))/np.std(g)
ga_s=(ga-np.mean(ga))/np.std(ga)
ppda_s=(ppda-np.mean(ppda))/np.std(ppda)
oppda_s=(oppda-np.mean(oppda))/np.std(oppda)
dc_s=(dc-np.mean(dc))/np.std(dc)
odc_s=(odc-np.mean(odc))/np.std(odc)
pts_s=(pts-np.mean(pts))/np.std(pts)
xg_s=(xg-np.mean(xg))/np.std(xg)
npxg_s=(npxg-np.mean(npxg))/np.std(npxg)
xga_s=(xga-np.mean(xga))/np.std(xga)
npxga_s=(npxga-np.mean(npxga))/np.std(npxga)
npxgd_s=(npxgd-np.mean(npxgd))/np.std(npxgd)

X_six = np.concatenate((g,ga,ppda,oppda,dc,odc),axis=1)
X_six_std = np.concatenate((g_s,ga_s,ppda_s,oppda_s,dc_s,odc_s),axis=1)

X_four_std=np.concatenate((ppda_s,oppda_s,dc_s,odc_s),axis=1)

X_six_exp = np.concatenate((X_six,xg,npxg,xga,npxga,npxgd),axis=1)
X_six_exp_std = np.concatenate((X_six_std,xg_s,npxg_s,xga_s,npxga_s,npxgd_s),axis=1)

X_more= np.concatenate((X_situation,X_gamestate,X_timing,X_shotZones,X_attack,X_result),axis=1)
scaler=StandardScaler()
scaler.fit(X_more)
X_more_std=scaler.transform(X_more)

X_SUPREMO_std= np.concatenate((X_six_std,X_more_std),axis=1)

X_and_Y=np.concatenate((X_SUPREMO_std,pts_s),axis=1)


#Lista che continiene i nomi di tutti gli attributi di X_SUPREMO
COLONNE = ['g', 'ga', 'ppda', 'oppda', 'dc', 'odc',
           'openplay_shots', 'fromcorner_shots', 'freekick_shots', 'setpiece_shots', 'penalty_shots',
           'openplay_goals', 'fromcorner_goals', 'freekick_goals', 'setpiece_goals', 'penalty_goals',
           'openplay_xG', 'fromcorner_xG', 'freekick_xG', 'setpiece_xG', 'penalty_xG',
           'openplay_shots_against', 'fromcorner_shots_against', 'freekick_shots_against', 'setpiece_shots_against','penalty_shots_against',
           'openplay_goals_against', 'fromcorner_goals_against', 'freekick_goals_against', 'setpiece_goals_against', 'penalty_goals_against',
           'openplay_xG_against', 'fromcorner_xG_against', 'freekick_xG_against', 'setpiece_xG_against', 'penalty_xG_against',
           'diff0_time', 'vantaggio1_time', 'vantaggio_multiplo_time', 'svantaggio1_time',
           'diff0_shots', 'vantaggio1_shots','vantaggio_multiplo_shots', 'svantaggio1_shots',
           'diff0_goals', 'vantaggio1_goals', 'vantaggio_multiplo_goals', 'svantaggio1_goals',
           'diff0_xG', 'vantaggio1_xG', 'vantaggio_multiplo_xG', 'svantaggio1_xG',
           'diff0_shots_against', 'vantaggio1_shots_against', 'vantaggio_multiplo_shots_against', 'svantaggio1_shots_against',
           'diff0_goals_against', 'vantaggio1_goals_against', 'vantaggio_multiplo_goals_against', 'svantaggio1_goals_against',
           'diff0_xG_against', 'vantaggio1_xG_against', 'vantaggio_multiplo_xG_against', 'svantaggio1_xG_against',
           'timing_1_15_shots', 'timing_16_30_shots', 'timing_31_45_shots', 'timing_46_50_shots', 'timing_61_75_shots', 'timing_76_plus_shots',
           'timing_1_15_goals', 'timing_16_30_goals', 'timing_31_45_goals', 'timing_46_50_goals', 'timing_61_75_goals', 'timing_76_plus_goals',
           'timing_1_15_xG', 'timing_16_30_xG', 'timing_31_45_xG', 'timing_46_50_xG', 'timing_61_75_xG', 'timing_76_plus_xG',
           'timing_1_15_shots_against', 'timing_16_30_shots_against', 'timing_31_45_shots_against', 'timing_46_50_shots_against', 'timing_61_75_shots_against', 'timing_76_plus_shots_against',
           'timing_1_15_goals_against', 'timing_16_30_goals_against', 'timing_31_45_goals_against', 'timing_46_50_goals_against', 'timing_61_75_goals_against', 'timing_76_plus_goals_against',
           'timing_1_15_xG_against', 'timing_16_30_xG_against', 'timing_31_45_xG_against', 'timing_46_50_xG_against', 'timing_61_75_xG_against', 'timing_76_plus_xG_against',
           'shot_zones_out_of_box__shots', 'shot_zones_penalty_area_shots', 'shot_zones_six_yard_box_shots',
           'shot_zones_out_of_box_goals','shot_zones_penalty_area_goals', 'shot_zones_six_yard_box_goals',
           'shot_zones_out_of_box_xG', 'shot_zones_penalty_area_xG', 'shot_zones_six_yard_box_xG',
           'shot_zones_out_of_box_shots_against', 'shot_zones_penalty_area_shots_against', 'shot_zones_six_yard_box_shots_against',
           'shot_zones_out_of_box_goals_against', 'shot_zones_penalty_area_goals_against', 'shot_zones_six_yard_box_goals_against',
           'shot_zones_out_of_box_xG_against', 'shot_zones_penalty_area_xG_against', 'shot_zones_six_yard_box_xG_against',
           'att_normal_shots', 'att_standard_shots', 'att_slow_shots', 'att_fast_shots',
           'att_normal_goals', 'att_standard_goals', 'att_slow_goals', 'att_fast_goals',
           'att_normal_xG', 'att_standard_xG', 'att_slow_xG', 'att_fast_xG',
           'att_normal_shots_against', 'att_standard_shots_against', 'att_slow_shots_against', 'att_fast_shots_against',
           'att_normal_goals_against', 'att_standard_goals_against', 'att_slow_goals_against', 'att_fast_goals_against',
           'att_normal_xG_against', 'att_standard_xG_against', 'att_slow_xG_against', 'att_fast_xG_against',
           'res_savedshot_shots', 'res_missedshot_shots', 'res_blockedshot_shots', 'res_goal_shots', 'res_shotonpost_shots',
           'res_savedshot_xG', 'res_missedshot_xG', 'res_blockedshot_xG', 'res_goalshot_xG', 'res_shotonpost_xG',
           'res_savedshot_xG_against', 'res_missedshot_xG_against', 'res_blockedshot_xG_against', 'res_goalshot_xG_against', 'res_shotonpost_xG_against']

COLONNE_arr=np.array(['g', 'ga', 'ppda', 'oppda', 'dc', 'odc',
           'openplay_shots', 'fromcorner_shots', 'freekick_shots', 'setpiece_shots', 'penalty_shots',
           'openplay_goals', 'fromcorner_goals', 'freekick_goals', 'setpiece_goals', 'penalty_goals',
           'openplay_xG', 'fromcorner_xG', 'freekick_xG', 'setpiece_xG', 'penalty_xG',
           'openplay_shots_against', 'fromcorner_shots_against', 'freekick_shots_against', 'setpiece_shots_against','penalty_shots_against',
           'openplay_goals_against', 'fromcorner_goals_against', 'freekick_goals_against', 'setpiece_goals_against', 'penalty_goals_against',
           'openplay_xG_against', 'fromcorner_xG_against', 'freekick_xG_against', 'setpiece_xG_against', 'penalty_xG_against',
           'diff0_time', 'vantaggio1_time', 'vantaggio_multiplo_time', 'svantaggio1_time',
           'diff0_shots', 'vantaggio1_shots','vantaggio_multiplo_shots', 'svantaggio1_shots',
           'diff0_goals', 'vantaggio1_goals', 'vantaggio_multiplo_goals', 'svantaggio1_goals',
           'diff0_xG', 'vantaggio1_xG', 'vantaggio_multiplo_xG', 'svantaggio1_xG',
           'diff0_shots_against', 'vantaggio1_shots_against', 'vantaggio_multiplo_shots_against', 'svantaggio1_shots_against',
           'diff0_goals_against', 'vantaggio1_goals_against', 'vantaggio_multiplo_goals_against', 'svantaggio1_goals_against',
           'diff0_xG_against', 'vantaggio1_xG_against', 'vantaggio_multiplo_xG_against', 'svantaggio1_xG_against',
           'timing_1_15_shots', 'timing_16_30_shots', 'timing_31_45_shots', 'timing_46_50_shots', 'timing_61_75_shots', 'timing_76_plus_shots',
           'timing_1_15_goals', 'timing_16_30_goals', 'timing_31_45_goals', 'timing_46_50_goals', 'timing_61_75_goals', 'timing_76_plus_goals',
           'timing_1_15_xG', 'timing_16_30_xG', 'timing_31_45_xG', 'timing_46_50_xG', 'timing_61_75_xG', 'timing_76_plus_xG',
           'timing_1_15_shots_against', 'timing_16_30_shots_against', 'timing_31_45_shots_against', 'timing_46_50_shots_against', 'timing_61_75_shots_against', 'timing_76_plus_shots_against',
           'timing_1_15_goals_against', 'timing_16_30_goals_against', 'timing_31_45_goals_against', 'timing_46_50_goals_against', 'timing_61_75_goals_against', 'timing_76_plus_goals_against',
           'timing_1_15_xG_against', 'timing_16_30_xG_against', 'timing_31_45_xG_against', 'timing_46_50_xG_against', 'timing_61_75_xG_against', 'timing_76_plus_xG_against',
           'shot_zones_out_of_box__shots', 'shot_zones_penalty_area_shots', 'shot_zones_six_yard_box_shots',
           'shot_zones_out_of_box_goals','shot_zones_penalty_area_goals', 'shot_zones_six_yard_box_goals',
           'shot_zones_out_of_box_xG', 'shot_zones_penalty_area_xG', 'shot_zones_six_yard_box_xG',
           'shot_zones_out_of_box_shots_against', 'shot_zones_penalty_area_shots_against', 'shot_zones_six_yard_box_shots_against',
           'shot_zones_out_of_box_goals_against', 'shot_zones_penalty_area_goals_against', 'shot_zones_six_yard_box_goals_against',
           'shot_zones_out_of_box_xG_against', 'shot_zones_penalty_area_xG_against', 'shot_zones_six_yard_box_xG_against',
           'att_normal_shots', 'att_standard_shots', 'att_slow_shots', 'att_fast_shots',
           'att_normal_goals', 'att_standard_goals', 'att_slow_goals', 'att_fast_goals',
           'att_normal_xG', 'att_standard_xG', 'att_slow_xG', 'att_fast_xG',
           'att_normal_shots_against', 'att_standard_shots_against', 'att_slow_shots_against', 'att_fast_shots_against',
           'att_normal_goals_against', 'att_standard_goals_against', 'att_slow_goals_against', 'att_fast_goals_against',
           'att_normal_xG_against', 'att_standard_xG_against', 'att_slow_xG_against', 'att_fast_xG_against',
           'res_savedshot_shots', 'res_missedshot_shots', 'res_blockedshot_shots', 'res_goal_shots', 'res_shotonpost_shots',
           'res_savedshot_xG', 'res_missedshot_xG', 'res_blockedshot_xG', 'res_goalshot_xG', 'res_shotonpost_xG',
           'res_savedshot_xG_against', 'res_missedshot_xG_against', 'res_blockedshot_xG_against', 'res_goalshot_xG_against', 'res_shotonpost_xG_against'])

ind_col_da_eliminare=[0,1,139, 5 ,12 , 13 ,119, 8, 104, 123, 30, 22, 56, 110, 131, 32, 113, 28, 35, 43, 47, 51, 36, 46, 39,
                     42, 60, 97, 98, 99, 101, 106, 147]

COLONNE_NUOVE=np.delete(COLONNE_arr,ind_col_da_eliminare,axis=0)

df_X_SUPREMO_std=pd.DataFrame(data=X_SUPREMO_std,columns=COLONNE)

#df_X_and_Y=pd.DataFrame(data=X_and_Y,columns=COLONNE+["Points"])
#df_X_and_Y.to_csv("Data/all_157+Points")


#Nota: (-1,17) impongo che il reshape avvenga a per forza 17 colonne e il numero di righe di conseguenza.
stats_champ_league=functionz.champions_team(lista_dataframe_complete).reshape(-1,17)
stats_retrocesse=functionz.retrocesse(lista_dataframe_complete).reshape(-1,17)

#CREO DATAFRAME CON LE SQUADRE IN CHAMPIONS
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
dataframe_champions=pd.DataFrame(data=data_champ,columns=["W","D","L","G", "GA", "PTS", "xG", "NPxG", "xGA",
                                                           "NPxGA", "NPxGD", "PPDA", "OPPDA", "DC", "ODC", "xPTS"])
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
dataframe_retro=pd.DataFrame(data=data_retro,columns=["W","D","L","G", "GA", "PTS", "xG", "NPxG", "xGA", 
                                                      "NPxGA", "NPxGD", "PPDA", "OPPDA", "DC", "ODC", "xPTS"])

#print(dataframe_retro)