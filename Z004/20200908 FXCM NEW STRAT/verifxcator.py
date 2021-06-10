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
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'



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
import configparser

engine = pyttsx3.init()
config = configparser.ConfigParser()

_total1 = 0

print('Librairies imported\n')

engine.say("librairie loaded")
engine.runAndWait()
print('Prêt')

_t1 = dt.datetime.now()

config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1') # 'm5'
_period2 = config.get('TIMEFRAME','_period2') # 'H1'
_period3 = config.get('TIMEFRAME','_period3') # 'D1'
_path1 = config.get('PATH','_path1') # 'Base/'
_path2 = config.get('PATH','_path2') # 'Base_Clean'
_path3 = config.get('PATH','_path3') # 'Base_Input'
TICKERS = config.get('TICKERS','TICKERS')
TICKERS = TICKERS.split(',')

_path = input('Quel path voulez-vous vérifier? (1, 2 ou 3')

if _path == 2:
    _path = _path2
elif _path == 3:
    _path = _path3
else:
    _path = _path1

def scrap_ini(x):
    try:
        globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period+'.csv')
        globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].set_index(globals()['df1_%s' %x.replace('/','')].Date,drop = True)
        globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].drop(['Date'],axis=1)
        print('\r',col.Fore.BLUE,'Vérification de la base',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    except:
        pass
    return(globals()['df1_%s' %x.replace('/','')])


def run_scrap():
    print('Scrap Ini')
    _t1 = dt.datetime.now()
    p = mp.Pool(os.cpu_count())
    p.map(scrap_ini,TICKERS) 
    p.close()
    p.join()
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()

def verif(x):
    globals()['LST1_%s' %x.replace('/','')] = list(globals()['df1_%s' %x.replace('/','')].index)
    globals()['PB1_%s' %x.replace('/','')] = [i for i, cnt in Counter(globals()['LST1_%s' %x.replace('/','')]).items() if cnt > 1]
    return(globals()['LST1_%s' %x.replace('/','')],globals()['PB1_%s' %x.replace('/','')],x)

def run_verif():
    print('Detection des problèmes...')
    _t1 = dt.datetime.now()
    p = mp.Pool(os.cpu_count())
    p.map(verif,TICKERS) 
    p.close()
    p.join()
    print('Fin du module détection pb')
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()


def method1(x):
    _total1 = 0
    print('\r',col.Fore.BLUE,'Calcul pour la paire',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    _total1 += len(globals()['PB1_%s' %x.replace('/','')])
    print('Total :',_total1)
    return()

def run_method1():
    print('Méthode 1')
    _t1 = dt.datetime.now()
    p=mp.Pool(os.cpu_count())
    p.map(method1,TICKERS) 
    p.close()
    p.join()
    _t2 = dt.datetime.now()
    print('Fin de Méthode 1')
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()

def method2(x):
    _total1 = 0
    print('\r',col.Fore.BLUE,'Calcul pour la paire',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    _total1 += (globals()['df1_%s' %x.replace('/','')].shape[0] - globals()['df1_%s' %x.replace('/','')].drop_duplicates().shape[0])   
    print('Total :',_total1)
    return()

def run_method2():
    print('\n Méthode 2')
    _t1 = dt.datetime.now()
    p=mp.Pool(os.cpu_count())
    p.map(method2,TICKERS) 
    p.close()
    p.join()
    _t2 = dt.datetime.now()
    print('Fin de Méthode 2')
    print("Temps d'excution du module",str((_t2 - _t1)))
    return

def calc_pb(x):
    _ct = 0
    print('\r',col.Fore.BLUE,'Calcul pour la paire',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    if globals()['df1_%s' %x.replace('/','')].drop_duplicates().shape[0] != globals()['df1_%s' %x.replace('/','')].shape[0]:
        print('probleme détecté dans', x)
        #print('\n',globals()['df1_%s' %x.replace('/','')].drop_duplicates())
        _ct += 1
    print('Au total,',_ct,'problème détecté')
    return()

def run_calcpb():
    print('Calcpb')
    _t1 = dt.datetime.now()
    p=mp.Pool(os.cpu_count())
    p.map(calc_pb,TICKERS) 
    p.close()
    p.join()
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()   

def calc_uni(x):
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
    return()

def run_calcuni():
    _t1 = dt.datetime.now()
    p=mp.Pool(os.cpu_count())
    p.map(calc_uni,TICKERS) 
    p.close()
    p.join()
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()

def run_main():
    _t1 = dt.datetime.now()
    change_tf()
    print(_period,_time_frame)
    run_scrap()
    run_verif()
    run_method1()
    run_method2()
    run_calcpb()
    run_calcuni()
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()

def change_tf():
    global _period,_time_frame
    _period = input('Entrez la timeframe (1, 2 ou 3')
    if _period == 2:
        _period = _period2
        _time_frame = 60
    elif _period == 3:
        _period = _period3
        _time_frame = 1440
    else:
        _period = _period1
        _time_frame = 5
    return(_period,_time_frame)

_t2 = dt.datetime.now()
print("Temps d'excution du module",str((_t2 - _t1)))

if __name__ == "__main__":
    from verifxcator import *
    run_main()

    