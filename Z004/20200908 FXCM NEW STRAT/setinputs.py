###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
############################      S E T T I N G       O F      T H E     I N P U T S        ###############################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'

import colorama as col
import pandas as pd
import talib
import os
import datetime as dt
from scraptickers import scrap_tickers
import pyttsx3
from sklearn.preprocessing import MinMaxScaler
import configparser
config = configparser.ConfigParser()

import configparser
config = configparser.ConfigParser()

scaler = MinMaxScaler(feature_range=(0, 1))

config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1') # 'm5'
_period2 = config.get('TIMEFRAME','_period2') # 'H1'
_period3 = config.get('TIMEFRAME','_period3') # 'D1'
_path1 = config.get('PATH','_path1') # 'Base/'
_path2 = config.get('PATH','_path2') # 'Base_Clean/'
_path3 = config.get('PATH','_path3') # 'Base_Input/'
_path4 = config.get('PATH','_path4') # 'Base_Signals/'
_path = _path2

engine = pyttsx3.init()
engine.say("Setting the in put")
engine.runAndWait()

def set_inputs(x):
    try:
        print('\r',col.Fore.BLUE,'Génération des inputs pour le ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
        #try:
        globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period1+'.csv')
        globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].set_index(globals()['df1_%s' %x.replace('/','')].Date,drop = True)
        globals()['df1_%s' %x.replace('/','')] = globals()['df1_%s' %x.replace('/','')].drop(['Date'],axis=1)

        df1 = globals()['df1_%s' %x.replace('/','')].copy()

        df1['delta_MM21_MM34'] = (talib.EMA(df1.Close, timeperiod=21) - talib.EMA(df1.CloseBid, timeperiod=34)) #* 1000
        df1['BOLUP'],df1['BOLMID'],df1['BOLLOW'] = talib.BBANDS(df1.CloseBid, timeperiod=252, nbdevup=2, nbdevdn=2, matype=0)
        df1['delta_price_BOLLOW'] = (df1.CloseBid - df1.BOLLOW) #* 1000
        df1['delta_price_BOLUP'] = (df1.CloseBid - df1.BOLUP) #* 1000
        df1['delat_price_ATR'] = (((df1.CloseBid - df1.OpenBid)) - talib.ATR(df1.HighBid,df1.LowBid,df1.CloseBid,timeperiod=5)) #* 1000
        df1['delta_high_low'] = (df1.HighBid - df1.LowBid) #* 1000
        df1['delta_close_open'] = (df1.CloseBid - df1.OpenBid) #* 1000

        df1['delta_MM21_MM34'] = scaler.fit_transform(df1[['delta_MM21_MM34']])
        df1['delta_price_BOLLOW'] = scaler.fit_transform(df1[['delta_price_BOLLOW']])
        df1['delta_price_BOLUP'] = scaler.fit_transform(df1[['delta_price_BOLUP']])
        df1['delat_price_ATR'] = scaler.fit_transform(df1[['delat_price_ATR']])
        df1['delta_high_low'] = scaler.fit_transform(df1[['delta_high_low']])
        df1['delta_close_open'] = scaler.fit_transform(df1[['delta_close_open']])

        df1 = df1.dropna()

        if os.path.isdir('Base_Input/') == False:
            os.makedirs('Base_Input/')

        df1.to_csv('Base_Input/'+x.replace('/','')+'m5.csv')
    except:
        print('')
        print(col.Fore.RED,'Problème avec le ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL)
        print(df1.info())
        print('')

    return()

engine.say("In put setted")
engine.runAndWait()

if __name__ == "__main__":
    pass