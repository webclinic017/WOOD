###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
##################################      C L E A N      B A S E      F X C M       #########################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'


import datetime as dt
import colorama as col
import pandas as pd
import multiprocessing.dummy as mp 
import os
import time
import configparser

config = configparser.ConfigParser()

config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1') # 'm5'
_period2 = config.get('TIMEFRAME','_period2') # 'H1'
_period3 = config.get('TIMEFRAME','_period3') # 'D1'
_period4 = config.get('TIMEFRAME','_period4') # 'W1'
_period5 = config.get('TIMEFRAME','_period5') # 'M1'
_period6 = config.get('TIMEFRAME','_period6') # 'm15'
_period7 = config.get('TIMEFRAME','_period7') # 'm30'
_period8 = config.get('TIMEFRAME','_period8') # 'H4'
_path1 = config.get('PATH','_path1') # 'Base/'
_path2 = config.get('PATH','_path2') # 'Base_Clean/'
_path3 = config.get('PATH','_path3') # 'Base_Input/'

TICKERS = ['EUR/USD']
x='EUR/USD'

'''
TO DO

globals()['df1_%s' %x.replace('/','')]['HighMax'] = globals()['df1_%s' %x.replace('/','')].HighBid.iloc[::-1].rolling(24).max().iloc[::-1]
globals()['df1_%s' %x.replace('/','')]['LowMin'] = globals()['df1_%s' %x.replace('/','')].LowBid.iloc[::-1].rolling(24).min().iloc[::-1]

'''

def scrap_ini(x):
   
    try:
        globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path1+x.replace('/','')+'_'+_period+'_BidAndAsk.csv',delimiter='|')
        print('\r',col.Fore.BLUE,'Reindexation de la base',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    
        globals()['df1_%s' %x.replace('/','')] = \
            globals()['df1_%s' %x.replace('/','')].set_index(pd.to_datetime(globals()['df1_%s' %x.replace('/','')].Date,format='%Y-%m-%d %H:%M:%S'),drop=True)
        globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].drop(['Date'],axis=1)
        globals()['df1_%s' %x.replace('/','')].columns = ['OpenBid', 'HighBid','LowBid','CloseBid','OpenAsk','HighAsk','LowAsk','CloseAsk','Total','None']
        globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].drop(['None'],axis=1)
        globals()['df1_%s' %x.replace('/','')]['OpenBid'] = \
        globals()['df1_%s' %x.replace('/','')]['OpenBid'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['HighBid'] = \
        globals()['df1_%s' %x.replace('/','')]['HighBid'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['LowBid'] = \
        globals()['df1_%s' %x.replace('/','')]['LowBid'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['CloseBid'] = \
        globals()['df1_%s' %x.replace('/','')]['CloseBid'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['OpenAsk'] = \
        globals()['df1_%s' %x.replace('/','')]['OpenAsk'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['HighAsk'] = \
        globals()['df1_%s' %x.replace('/','')]['HighAsk'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['LowAsk'] = \
        globals()['df1_%s' %x.replace('/','')]['LowAsk'].replace(to_replace=',',value='.',regex=True).astype(float)
        globals()['df1_%s' %x.replace('/','')]['CloseAsk'] = \
        globals()['df1_%s' %x.replace('/','')]['CloseAsk'].replace(to_replace=',',value='.',regex=True).astype(float)

        globals()['df1_%s' %x.replace('/','')]['Open'] = (globals()['df1_%s' %x.replace('/','')].OpenBid + globals()['df1_%s' %x.replace('/','')].OpenAsk)/2
        globals()['df1_%s' %x.replace('/','')]['High'] = (globals()['df1_%s' %x.replace('/','')].HighBid + globals()['df1_%s' %x.replace('/','')].HighAsk)/2
        globals()['df1_%s' %x.replace('/','')]['Low'] = (globals()['df1_%s' %x.replace('/','')].LowBid + globals()['df1_%s' %x.replace('/','')].LowAsk)/2
        globals()['df1_%s' %x.replace('/','')]['Close'] = (globals()['df1_%s' %x.replace('/','')].CloseBid + globals()['df1_%s' %x.replace('/','')].CloseAsk)/2
        if _period == 'm5':
            globals()['df1_%s' %x.replace('/','')]['HigMax'] = \
                globals()['df1_%s' %x.replace('/','')]['HighBid'].iloc[::-1].rolling(99).max().iloc[::-1]
            globals()['df1_%s' %x.replace('/','')]['LowMin'] = \
                globals()['df1_%s' %x.replace('/','')]['LowAsk'].iloc[::-1].rolling(99).min().iloc[::-1]


        globals()['df1_%s' %x.replace('/','')].to_csv(_path2+x.replace('/','')+_period+'.csv')
    except:
        print('Problème pour le ticker',x,'en timeframe',_period)
    return()

def clean_base():
    global _period
    _t1 = dt.datetime.now()
    for _period in [_period1,_period2,_period3,_period4,_period5,_period6,_period7,_period8]:
        print('Début des opérations pour la timeframe',_period)
        p=mp.Pool(os.cpu_count())
        p.map(scrap_ini,TICKERS) 
        p.close()
        p.join()
        print("\n\n ===> Opération terminée. Tout est nettoyé et prêt à l'utilisation pour la time frame",_period)
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()

if __name__ == "__main__":
    pass