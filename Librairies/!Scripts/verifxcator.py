###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
##########################################      V E R I F X C A T O R      ################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

############################
######## LIBRAIRIES ########
############################
print('Importing Librairies...')
import clearall
import colorama as col
import pandas as pd
import datetime as dt
import time
import os
import datetime as dt
import numpy as np
import multiprocessing.dummy as mp
from collections import Counter
import pyttsx3

engine = pyttsx3.init()

print('Librairies imported\n')

engine.say("librairie loaded")
engine.runAndWait()
print('Prêt')

TICKERS = [ 'EUR/USD','USD/JPY','GBP/USD','USD/CHF','EUR/CHF','AUD/USD','USD/CAD','NZD/USD','EUR/GBP','EUR/JPY','GBP/JPY','CHF/JPY',\
'GBP/CHF','EUR/AUD','EUR/CAD','AUD/CAD','AUD/JPY','CAD/JPY','NZD/JPY','GBP/CAD','GBP/NZD','GBP/AUD','AUD/NZD','USD/SEK','EUR/SEK',\
'EUR/NOK','USD/NOK','USD/MXN','AUD/CHF','EUR/NZD','USD/ZAR','ZAR/JPY','NZD/CHF','CAD/CHF','NZD/CAD','USD/ILS','USD/CNH']


_period = input('Entrez la timeframe (m5 - m15 - m30 - H1 - H4 - D1 - M1 - W1')

_time_frame = int(input("Entrez l'écart (5 - 15 - 30 - 60 - 240 - 1440"))

def scrap_ini(x):
    #_stop1 = (dt.datetime.now() - dt.timedelta(days=1)).strftime(format='%Y-%m-%d %23:00:00')
    #_start1 = (dt.datetime.now() - _delta_m5).strftime('%Y-%m-%d %H:%M:00')
    
    globals()['df1_%s' %x.replace('/','')] = pd.read_csv('Base/'+x.replace('/','')+'_'+_period+'_BidAndAsk.csv')
    print('\r',col.Fore.BLUE,'Reindexation de la base',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    INDEX = []
    for i in range(len(globals()['df1_%s' %x.replace('/','')].Date)):
        INDEX.append(pd.to_datetime((globals()['df1_%s' %x.replace('/','')].Date[i][6:]+'-'+globals()['df1_%s' %x.replace('/','')].Date[i][:2]+\
        '-'+globals()['df1_%s' %x.replace('/','')].Date[i][3:5]+' '+globals()['df1_%s' %x.replace('/','')].Time[i]),format='%Y-%m-%d %H:%M:%S'))

    globals()['df1_%s' %x.replace('/','')].index = INDEX
    globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].drop(['Date'],axis=1)
    globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].drop(['Time'],axis=1)   
    
    return([globals()['df1_%s' %x.replace('/','')] for x in TICKERS])#,_stop1)


    for x in TICKERS:
        globals()['df1_%s' %x.replace('/','')] = pd.DataFrame()


p=mp.Pool(os.cpu_count())
p.map(scrap_ini,TICKERS) 
p.close()
p.join()


print('Detection des problèmes...')
for x in TICKERS:
    globals()['LST1_%s' %x.replace('/','')] = list(globals()['df1_%s' %x.replace('/','')].index.strftime('%Y-%m-%d %H:%M:%S'))
    globals()['PB1_%s' %x.replace('/','')] = [i for i, cnt in Counter(globals()['LST1_%s' %x.replace('/','')]).items() if cnt > 1]


_total1 = 0
print('Méthode 1')
for x in TICKERS:
    print('\r',col.Fore.BLUE,'Calcul pour la paire',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    _total1 += len(globals()['PB1_%s' %x.replace('/','')])
   
print('Total :',_total1)

_total1 = 0

for x in TICKERS:
    print('\n Méthode 2')
    print('\r',col.Fore.BLUE,'Calcul pour la paire',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    _total1 += (globals()['df1_%s' %x.replace('/','')].shape[0] - globals()['df1_%s' %x.replace('/','')].drop_duplicates().shape[0])
   
print('Total :',_total1)

_ct = 0
for x in TICKERS:
    print('\r',col.Fore.BLUE,'Calcul pour la paire',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    if globals()['df1_%s' %x.replace('/','')].drop_duplicates().shape[0] != globals()['df1_%s' %x.replace('/','')].shape[0]:
        print('probleme détecté dans', x)
        print(globals()['df1_%s' %x.replace('/','')].drop_duplicates())
        _ct += 1
   
print('Au total,',_ct,'problème détecté')    

for x in TICKERS:
    globals()['_ct_%s' %x.replace('/','')] = 0
    print(col.Fore.BLUE,'Calcul des problèmes par ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL)
    globals()['_ct_%s' %x.replace('/','')] = 0
    for loop in range(1,len(globals()['df1_%s' %x.replace('/','')])):
        _delta = pd.to_datetime(globals()['df1_%s' %x.replace('/','')].index[loop]) - pd.to_datetime(globals()['df1_%s' %x.replace('/','')].index[loop-1])
        if  _delta != dt.timedelta(minutes=_time_frame) and _delta <= dt.timedelta(hours = 3):
            globals()['_ct_%s' %x.replace('/','')] += 1
            print(globals()['df1_%s' %x.replace('/','')].index[loop-1],'---',globals()['df1_%s' %x.replace('/','')].index[loop],'---',globals()['df1_%s' %x.replace('/','')].index[loop+1],' -- ',_delta)
    print('Pour le ticker',x,'il y a eu',globals()['_ct_%s' %x.replace('/','')],'problèmes')
    print('Soit',round(globals()['_ct_%s' %x.replace('/','')]*100/len(globals()['df1_%s' %x.replace('/','')]),2),'%')


def write_csv(x):
    globals()['df1_%s' %x.replace('/','')].to_csv('Base_Clean/'+['df1_%s' %x.replace('/','')]+'_'+_period+'.csv')




    