###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
############################      G E N E R A T I O N      O F      S I G N A L S         #################################
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
from sklearn.linear_model import LinearRegression
import colorama as col
import pandas as pd
import time
import os
import datetime as dt
import numpy as np
import shelve
import talib
import statistics
import pyttsx3
import configparser

engine = pyttsx3.init()
config = configparser.ConfigParser()

print('Librairies imported\n')

config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1') # 'm5'
_period2 = config.get('TIMEFRAME','_period2') # 'H1'
_period3 = config.get('TIMEFRAME','_period3') # 'D1'

_path1 = config.get('PATH','_path1') # 'Base/'
_path2 = config.get('PATH','_path2') # 'Base_Clean/'
_path3 = config.get('PATH','_path3') # 'Base_Input/'
_path4 = config.get('PATH','_path4') # 'Base_Signals/'
_path5 = config.get('PATH','_path5') # 'Base_Prepared/'
_path = _path3

_target = float(config.get('PARAMETRES','_target')) # take profit 0.0020
_sl = float(config.get('PARAMETRES','_sl')) # stop loss 0.0020
_spread_filter = float(config.get('PARAMETRES','_spread_filter')) # filtre spread 0.00050

def gen_sig(x):
    
    print('\r',col.Fore.BLUE,'Génération des signaux pour le ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL,end='',flush=True)
    globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period1+'.csv')

    try:
        globals()['df1_%s' %x.replace('/','')].drop(['HigMax','LowMin'],axis=1)
    except:
        pass

    globals()['df1_%s' %x.replace('/','')]['HigMax'] = globals()['df1_%s' %x.replace('/','')]['HighBid'].iloc[::-1].rolling(25).max().iloc[::-1] # inversion ordre
    globals()['df1_%s' %x.replace('/','')]['LowMin'] = globals()['df1_%s' %x.replace('/','')]['LowAsk'].iloc[::-1].rolling(25).min().iloc[::-1] # inversion ordre

    globals()['df1_%s' %x.replace('/','')].iloc[:]['HigMax']

    df1 = globals()['df1_%s' %x.replace('/','')].copy()

    MM21 = talib.EMA(df1.Close, timeperiod=21)
    rsi = talib.RSI(df1.Close, timeperiod=14)
    rsi_high = 65
    rsi_low = 35
    upperband, middleband, lowerband = talib.BBANDS(df1.Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    fastk, fastd = talib.STOCHRSI(df1.Close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)#STOCHASTICRSI
    slowk, slowd = talib.STOCH(df1.High, df1.Low, df1.Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)#STOCHASTIC


    df1['Signal'] = np.where(((slowk.shift(1) > 80) & (slowk.shift(1) > slowd.shift(1)) & (slowk < slowd) & (df1.High.shift(1) > df1.High.shift(2)) \
        & (df1.High < df1.High.shift(2))),-1,\
            np.where((slowk.shift(1) < 20) & (slowk.shift(1) < slowd.shift(1)) & (slowk > slowd) & \
                (df1.Low.shift(1) < df1.Low.shift(2)) & (df1.Low > df1.Low.shift(2)),1,0))

    df1['BUY'] = np.where((df1['HigMax'] > df1['CloseAsk'] * (1 + _target)) & (df1['LowMin'] >  df1['CloseAsk'] * (1 - _sl)),1,0)
    df1['SELL'] = np.where((df1['LowMin'] < df1['CloseBid'] * (1 - _target)) & (df1['HigMax'] <  df1['CloseBid'] * (1 + _sl)),1,0)
    
    
    #df1 = df1.drop(['OpenBid', 'HighBid', 'LowBid', 'OpenAsk', 'HighAsk', 'LowAsk',\
    #                'Total', 'Open', 'High', 'Low', 'Close', 'BOLUP', 'BOLMID', 'BOLLOW','CloseAsk',\
    #                'delta_price_BOLLOW','delta_price_BOLUP','delat_price_ATR','delta_high_low','delta_close_open'],axis=1)
    #df1 = df1[['08:30:00'<df1.Date.iloc[i][11:19]<'17:00:00' for i in range(len(df1))]]            
    df1 = df1[df1.Signal != 0]
    df1.to_csv(_path4+x.replace('/','')+'m5.csv')
    return()

engine.say("Generating the signauls")
engine.runAndWait()

if __name__ == "__main__":
    pass