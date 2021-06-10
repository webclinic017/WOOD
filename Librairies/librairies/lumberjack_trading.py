__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'
__version__ = 'v0.1'

####################################################################
####################################################################
############################### TRADING ############################
####################################################################
####################################################################

'''
tool for algotrading
'''

# Math Stuff
import numpy as np
import pandas as pd
import scipy.stats as stat

# Proceesing librairies
import keras
from natsort import natsorted
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve,roc_auc_score
import imblearn

# Side Stuff
import joblib
import warnings
import colorama as col
import pyttsx3
engine = pyttsx3.init()
from tqdm import tqdm, tqdm_notebook, tqdm_pandas
from functools import reduce

# Random Seed
seed_value = 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


# Technical analysis
from finta import TA
from ta.momentum import AwesomeOscillatorIndicator,KAMAIndicator, ROCIndicator,RSIIndicator,StochRSIIndicator, TSIIndicator, WilliamsRIndicator
from ta.volatility import AverageTrueRange, DonchianChannel
from ta.trend import AroonIndicator, ADXIndicator
import talib

# Plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# Time Stuff
import time
import datetime as dt

print('Imblearn version ',imblearn.__version__)
print('Pandas version ',pd.__version__)
print('Numpy version ',np.__version__)
print('Tensorflow version ',tf.__version__)
print('Joblib version ',joblib.__version__)

warnings.filterwarnings('ignore')

TIK = ['AUD','NZD','GBP','JPY','CHF','CAD','SEK','NOK','ILS','MXN','USD','EUR']
RATE = [0.776,0.721,1.3912,1/105.91,1/0.892,1/1.2681,1/8.2884,1/8.4261,1/3.2385,1/20.1564,1,1.21]
df_ratefx = pd.DataFrame(index=TIK)
df_ratefx['rate'] = RATE

# BACKTEST
def bt(price,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
        _rate,x,_target,_exposure,_size,_sl,_save=0,_bt_report=0,_trigger_rsi=0,_trigger_spread=0.025,_period='m5',_period2='H1'):
    engine = pyttsx3.init()
    _ticker = x.replace('/','')
    if _verbose !=0 : 
        print(col.Fore.YELLOW)
        print('\n_________________________________________________\n')
        print('__________________________________')
        print('     ___ Period 1 : => ', _period,' ___')
        print('     ___ Period 2 : => ', _period2,' ___')
        print('__________________________________')
        print('Ca$h Ini :',_cash_ini)
        print('Date début:',_year_bottom)
        print('Date fin :',_year_top)
        print('Etat verbose :',_verbose)
        print('Nombre Bougies Exit :',_nb_bougie_exit)
        print('Target :',_target)
        print('Stop Loss :',_sl)
        print('Trigger Reengage :',_trigger_reengage)
        print('Trigger Target :',_trigger_target)
        print('Trigger Inverse :',_trigger_invers)
        print('Trigger StopLoss :',_trigger_sl)
        print('Trigger RSI :',_trigger_rsi)
        print('Exposure :',_exposure)
        print('_bt_report : ',_bt_report)
        print('Verbose : ',_verbose)
        print('Trigger Spread :',_trigger_spread)
        print('Save : ',_save)
        print('\n_________________________________________________\n')
        print(col.Style.RESET_ALL)

    engine.say("Backtesting in progress")
    engine.runAndWait()


    _t1 = dt.datetime.now()
    print('Début des opérations horodatée à',dt.datetime.now())
    print('_trigger_rsi :',_trigger_rsi)

    _total = 0
    _cash = _cash_ini
    _pnl = 0
    _tracker = 0

    DATE = []
    CONTRACT = []
    OPEN_POZ = []
    CLOSE_POZ = []
    RATE_OPEN_POZ = []
    RATE_CLOSE_POZ = []
    PNL_LAT = []
    PNL_REAL = []
    TOTAL_OPEN = []
    TOTAL_CLOSE = []
    PRICE_BUY = []
    PRICE_SELL = []
    DER_POZ = []
    TOTAL_PNL_LAT = []
    TOTAL_PNL_REAL = []
    EXPO_MAX = []
    TRACKER = []
    TRADE_DURATION = []
    PRICE_BUY = []
    PRICE_SELL = []
    EQUITY = []
    CASH = []
    
    df_resultats = pd.DataFrame(index=['Equity','Nbre Winners','Nbre winners long','Nbre winners short','Nbre Loosers','Nbre loosers long','Nbre loosers short','Max lenght of trade','Min lenght of trade',\
        'Average lenght of trade','Cumul pnl'])
    
    engine.say("קדימה")
    engine.runAndWait()

    if _verbose !=0 :
        print('\nChargement de la nouvelle base\n\n')
        print(col.Fore.MAGENTA,'Le rate du ticker',x,'est à ',_rate,col.Style.RESET_ALL)

        print('Bases chargées')

        print('TETEL process effectué')

        print(col.Fore.CYAN,'ENTERING THE BACKTEST',col.Style.RESET_ALL)

    price = price[(price.index >= _year_bottom) & (price.index <= _year_top)]
    time.sleep(0.2)
        
    price = price.dropna()

    _position = 0
    _equity = 0
    _nbtransactions = 0
    backtest_graph = pd.DataFrame()

    _winner = 0
    _looser = 0
    _longwinner = 0
    _longlooser = 0
    _shortwinner = 0
    _shortlooser = 0
    _index_entry = 0
    
    _average_duration = 0
    _total = 0

    

    _open_buy = 0
    _open_sell = 0

    for i in tqdm(range(0,len(price))):

        _size = _cash_ini / df_ratefx.loc[x[:3],'rate']

        ##### POSITIONS EN L'AIR 
        if i >= (len(price)-1) and (_position == 1 or _position == -1) :

            if _position == -1:
                _position = 99
                _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _total += _pnl
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                
                if _pnl > 0:
                    _winner += _open_sell
                    _longwinner+=_open_sell
                else:
                    _looser += _open_sell
                    _shortlooser +=_open_sell
                
                TRADE_DURATION.append(i - _index_entry)
                
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    print(col.Fore.CYAN,"Cloture des positions en l'air",col.Style.RESET_ALL)
                    print(_open_sell,'position closed at',price.CloseAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(-1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseAsk.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_sell)
                PRICE_SELL = []
                _open_sell = 0
                continue

            if _position == 1:

                _position = 99
                _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _total += _pnl
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                
                if _pnl > 0:
                    _winner += _open_buy
                    _longwinner +=_open_buy
                else:
                    _looser += _open_buy
                    _longlooser += _open_buy

                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    print(col.Fore.CYAN,"Cloture des positions en l'air",col.Style.RESET_ALL)
                    print(_open_buy,'positions closed at',price.CloseBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseBid.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_buy) 
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                PRICE_BUY = []
                _open_buy = 0
                continue
        
        # SI PAS DE POSITION
        if _position == 0:
            # BUY SIGNAL
            if  price.Signal[i] == 1: 
                _pnl = 0
                _open_buy += 1
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                _position = 1
                _index_entry = i
                _tracker = price.index[i]
                _nbtransactions += 1
                price_buy = price.CloseAsk.iloc[i]
                PRICE_BUY.append(price_buy)
                _price_buy_mean = round(sum(PRICE_BUY)/len(PRICE_BUY),5)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    print('Position 1 bought at', price_buy,'(verification liste',PRICE_BUY[-1],')')

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(1)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(price_buy)
                RATE_CLOSE_POZ.append(0)
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_OPEN.append(1) 
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0)
                continue 

            # SELL SIGNAL
            elif price.Signal[i] == -1: 
                _pnl = 0
                _open_sell += 1
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                _index_entry = i
                _tracker = price.index[i]
                _position = -1
                _nbtransactions += 1
                price_sell = price.CloseBid.iloc[i]
                PRICE_SELL.append(price_sell)
                _price_sell_mean = round(sum(PRICE_SELL)/len(PRICE_SELL),5)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    print('Position 1 sold at', price_sell,'(verification liste',PRICE_SELL[-1],')')

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(-1)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(price_sell)
                RATE_CLOSE_POZ.append(0) 
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0)
                TOTAL_OPEN.append(1)
                continue

            else :
                _pnl = 0
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                PNL_LAT.append(0)
                PNL_REAL.append(0)
                continue
        
        # SI POSITION LONG
        elif _position == 1:

            ### RE_ENGAGE BUY ON VALID SIGNAL
            if price.Signal[i] == 1 and i - _index_entry < _nb_bougie_exit and _trigger_reengage == 1\
                 and _open_buy < _exposure :
                _pnl = 0
                _open_buy += 1
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                _position = 1
                _index_entry = i
                _tracker = price.index[i]
                _nbtransactions += 1
                price_buy = price.CloseAsk.iloc[i]
                PRICE_BUY.append(price_buy)
                _price_buy_mean = round(sum(PRICE_BUY)/len(PRICE_BUY),5)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    print('Position (REENG) 1 bought at', price_buy,'(verification liste',PRICE_BUY[-1],')')

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(1)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(price_buy)
                RATE_CLOSE_POZ.append(0)
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_OPEN.append(1) 
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0)
                continue
            
            ### CLOSE LONG ON RSI DROUP OUT
            if _trigger_rsi == 1 and price.MYRSI_2[i] > 85:
                _position = 0
                _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_buy)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_buy
                    _longwinner += _open_buy
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_buy
                    _longlooser +=_open_buy

                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0:
                        print(_open_buy,'positions (RSI) closed at',price.CloseBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else :
                        print(_open_buy,'positions (RSI) closed at',price.CloseBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseBid.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_buy) 
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                PRICE_BUY = []
                _open_buy = 0
                continue

            ### CLOSE LONG ON INVERSE SIGNAL
            if price.Signal[i] == -1 and _trigger_invers == 1:
                _position = 0
                _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_buy)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_buy
                    _longwinner += _open_buy
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_buy
                    _longlooser +=_open_buy

                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0:
                        print(_open_buy,'positions (INV) closed at',price.CloseBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else :
                        print(_open_buy,'positions (INV) closed at',price.CloseBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseBid.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_buy) 
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                PRICE_BUY = []
                _open_buy = 0
                continue
            
            ### CLOSE LONG ON TIME EXIT
            if i - _index_entry >= _nb_bougie_exit:
                _position = 0
                _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_buy)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_buy
                    _longwinner +=_open_buy
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_buy
                    _longlooser += _open_buy

                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0:
                        print(_open_buy,'positions (TIME EXIT) closed at',price.CloseBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else :
                        print(_open_buy,'positions (TIME EXIT) closed at',price.CloseBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseBid.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_buy) 
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                PRICE_BUY = []
                _open_buy = 0
                continue
            
            # CLOSE LONG ON TARGET
            if (float(price.HighBid.iloc[i]) - float(_price_buy_mean))/float(_price_buy_mean) >= _target and _trigger_target == 1:
                _position = 0
                _pnl = (price.HighBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_buy)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_buy
                    _longwinner += _open_buy
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_buy
                    _longlooser += _open_buy

                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0:
                        print(_open_buy,'positions (TG) closed at',price.HighBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else :
                        print(_open_buy,'positions (TG) closed at',price.HighBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.HighBid.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_buy) 
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                PRICE_BUY = []
                _open_buy = 0
                continue

            # CLOSE LONG ON STOP LOSS
            if (float(price. LowBid.iloc[i]) - float(_price_buy_mean))/float(_price_buy_mean) <= - _sl and _trigger_sl == 1:
                _position = 0
                _pnl = (price.LowBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_buy)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_buy
                    _longwinner += _open_buy
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_buy
                    _longlooser += _open_buy

                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0:
                        print(_open_buy,'positions (SL) closed at',price.LowBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else :
                        print(_open_buy,'positions (SL) closed at',price.LowBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.LowBid.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_buy) 
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                PRICE_BUY = []
                _open_buy = 0
                continue
            
            else:

                _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy * _rate
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(0) 
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0)
                continue 
 
        # SI POSITION SHORT
        elif _position == -1:

            ### RE-ENGAGE SELL ON VALID SIGNAL
            if price.Signal[i] == -1 and i - _index_entry < _nb_bougie_exit and _trigger_reengage == 1 \
                and _open_sell < _exposure :
                
                _pnl = 0
                _open_sell += 1
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)
                _index_entry = i
                _tracker = price.index[i]
                _position = -1
                _nbtransactions += 1
                price_sell = price.CloseBid.iloc[i]
                PRICE_SELL.append(price_sell)
                _price_sell_mean = round(sum(PRICE_SELL)/len(PRICE_SELL),5)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    print('Position (REENG) 1 sold at', price_sell,'(verification liste',PRICE_SELL[-1],')')

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(-1)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(price_sell)
                RATE_CLOSE_POZ.append(0) 
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0)
                TOTAL_OPEN.append(1)
                continue

            ### CLOSE SHORT ON RSI DROUP OUT
            if _trigger_rsi == 1 and price.MYRSI_2[i] < 15:   
                _position = 0
                _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_sell)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_sell
                    _shortwinner += _open_sell
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_sell
                    _shortlooser += _open_sell
                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0 :    
                        print(_open_sell,'position (RSI) closed at',price.CloseAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else:
                        print(_open_sell,'position (RSI) closed at',price.CloseAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(-1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseAsk.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_sell)
                PRICE_SELL = []
                _open_sell = 0
                continue

            ### CLOSE SHORT ON INVERSE SIGNAL
            if price.Signal[i] == 1 and _trigger_invers == 1:   
                _position = 0
                _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_sell)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_sell
                    _shortwinner += _open_sell
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_sell
                    _shortlooser += _open_sell
                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0 :    
                        print(_open_sell,'position (INV) closed at',price.CloseAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else:
                        print(_open_sell,'position (INV) closed at',price.CloseAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(-1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseAsk.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_sell)
                PRICE_SELL = []
                _open_sell = 0
                continue

            ### CLOSE SHORT ON TIME EXIT
            if i - _index_entry >= _nb_bougie_exit:   
                _position = 0
                _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_sell)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_sell
                    _shortwinner += _open_sell
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_sell
                    _shortlooser += _open_sell
                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0 :    
                        print(_open_sell,'position (TIME EXIT) closed at',price.CloseAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else:
                        print(_open_sell,'position (TIME EXIT) closed at',price.CloseAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(-1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.CloseAsk.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_sell)
                PRICE_SELL = []
                _open_sell = 0
                continue

            ### CLOSE SHORT ON TARGET
            if (float(price.LowAsk.iloc[i]) - float(_price_sell_mean))/float(_price_sell_mean) <= -_target and _trigger_target == 1:
                _position = 0
                _pnl = - (price.LowAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_sell)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_sell
                    _shortwinner += _open_sell
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_sell
                    _shortlooser +=_open_sell
                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0 :    
                        print(_open_sell,'position (TG) closed at',price.LowAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else:
                        print(_open_sell,'position (TG) closed at',price.LowAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(-1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.LowAsk.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_sell)
                PRICE_SELL = []
                _open_sell = 0
                continue

            ### CLOSE SHORT ON STOP LOSS
            if (float(price.HighAsk.iloc[i]) - float(_price_sell_mean))/float(_price_sell_mean) > _sl and _trigger_sl == 1:
                _position = 0
                _pnl = - (price.HighAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _total += _pnl
                _cash += _pnl
                _equity = _cash
                EQUITY.append(_equity)
                EXPO_MAX.append(_open_sell)
                CASH.append(_cash)
                if _pnl >=0:
                    _winner += _open_sell
                    _shortwinner += _open_sell
                    TRACKER.append(_tracker)
                else:
                    _looser += _open_sell
                    _shortlooser +=_open_sell
                TRADE_DURATION.append(i - _index_entry)
                if _verbose == 2:
                    print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                    if _pnl < 0 :    
                        print(_open_sell,'position (SL) closed at',price.HighAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                    else:
                        print(_open_sell,'position (SL) closed at',price.HighAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
                    print('nombre de candles en position :',i - _index_entry)
                    print('Equity :', _equity)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(-1)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(price.HighAsk.iloc[i])
                PNL_LAT.append(0)
                PNL_REAL.append(_pnl)
                TOTAL_PNL_LAT.append(0)
                TOTAL_PNL_REAL.append(_pnl)
                TOTAL_CLOSE.append(_open_sell)
                PRICE_SELL = []
                _open_sell = 0
                continue

            else:

                _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell * _rate
                _equity = _cash + _pnl

                EQUITY.append(_equity)
                CASH.append(_cash)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(0)
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0)
                continue
        

    try:
        _average_duration = round(sum(TRADE_DURATION)/len(TRADE_DURATION),2)
        _max_duration = max(TRADE_DURATION)
        _min_duration = min([item for item in TRADE_DURATION if item !=0])

    except:
        if _verbose != 0:
            print("(No Duration)") 
        _average_duration = 'NA'
        _max_duration = 0.00002
        _min_duration = 0.00001 
    if _verbose != 0:
        print(col.Fore.BLUE,'For ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL)
        if _total > 0:              
            print(col.Fore.MAGENTA,"\nTotal Profit & Loss : $",col.Fore.GREEN,round(_total,2),'. En ',\
                _nbtransactions,col.Style.RESET_ALL,' transactions.' )
        else:
            print(col.Fore.MAGENTA,"\nTotal Profit & Loss : $",col.Fore.RED,round(_total,2),'. En ',\
                _nbtransactions,col.Style.RESET_ALL,' transactions.' ) 
        print(col.Fore.GREEN,"\nWinners Number :",_winner,col.Style.RESET_ALL)
        print(col.Fore.RED,"\nLoosers number :",_looser,col.Style.RESET_ALL)

    backtest_graph['Equity'] = EQUITY

    df_resultats[x] = [(round(_equity,2)),(_winner),(_longwinner),(_shortwinner),(_looser),(_longlooser),(_shortlooser),(_max_duration),(_min_duration),(_average_duration),(_total)]

    DER_POZ.append(_pnl)

    engine.say("Finito caucau")
    engine.runAndWait()
    _t2 = dt.datetime.now()
    print("BT's execution time",str((_t2 - _t1)))
    df_historical = pd.DataFrame()
    df_historical = pd.DataFrame(index=DATE)
    df_historical['Contract'] = CONTRACT
    df_historical['Open_Poz'] = OPEN_POZ
    df_historical['Close_Pos'] = CLOSE_POZ
    df_historical['Rate_Open_Poz'] = RATE_OPEN_POZ 
    df_historical['Rate_Close_Poze'] = RATE_CLOSE_POZ
    df_historical['Pnl_Lat'] = TOTAL_PNL_LAT
    df_historical['Pnl_Real'] = TOTAL_PNL_REAL
    df_historical = df_historical.sort_index()
    _generated_cash = round(df_historical.Pnl_Real.sum(),2)
    _generated_cash_perc = round((_generated_cash / _cash_ini) * 100,2)
    if _verbose != 0:
        print(col.Fore.YELLOW,x,col.Fore.BLUE,'results',col.Style.RESET_ALL)
        print(col.Fore.MAGENTA,'Tested Period',_year_bottom,' à',_year_top,col.Style.RESET_ALL)
    print(col.Fore.CYAN,'Total Number of trades',max([sum(TOTAL_OPEN),sum(TOTAL_CLOSE)]),col.Style.RESET_ALL)
    if _verbose != 0:
        if _generated_cash <= 0:
            print('Started Cash :',_cash_ini)
            print('P&L in currency:',col.Fore.RED,str(_generated_cash)+'$',col.Style.RESET_ALL)
            print('P&L in %:',col.Fore.RED,str(_generated_cash_perc)+'%',col.Style.RESET_ALL)

        else:
            print('Started Cash :',_cash_ini)
            print('P&L  in currency:',col.Fore.GREEN,str(_generated_cash)+'$',col.Style.RESET_ALL)
            print('P&L in %:',col.Fore.GREEN,str(_generated_cash_perc)+'%',col.Style.RESET_ALL)

        print('Average trade duration',_average_duration)
        print('# Winners ',df_resultats.T['Nbre Winners'].sum())
        print('# Winners long ',df_resultats.T['Nbre winners long'].sum())
        print('# Winners short ',df_resultats.T['Nbre winners short'].sum())

        print('# Loosers ',df_resultats.T['Nbre Loosers'].sum())
        print('# Loosers  long',df_resultats.T['Nbre loosers long'].sum())
        print('# Loosers  short',df_resultats.T['Nbre loosers short'].sum())
        print('Cumulated gains',round(df_historical[df_historical.Pnl_Real>0].Pnl_Real.sum(),2))
        print('Cumulated losses',round(df_historical[df_historical.Pnl_Real<0].Pnl_Real.sum(),2))
    print(col.Fore.BLUE,'PROFIT FACTOR : ',\
            abs(round(df_historical[df_historical.Pnl_Real>0].Pnl_Real.sum()/df_historical[df_historical.Pnl_Real<0].Pnl_Real.sum(),2)),col.Style.RESET_ALL)
    try:
        print(col.Fore.CYAN,'Winners Ratio :',\
            round((df_resultats.T['Nbre Winners'].sum()*100)/(df_resultats.T['Nbre Loosers'].sum()+df_resultats.T['Nbre Winners'].sum()),2),\
                '%',col.Style.RESET_ALL)
    except:
        print(col.Fore.CYAN,'Winners Ratio  :None',col.Style.RESET_ALL)
    if _verbose != 0:
        try:
            print('Average Winners',round(sum(list(filter(lambda x:  x > 0,PNL_REAL)))/len(list(filter(lambda x:  x > 0,PNL_REAL))),2))
            print('% Average Winners',round(sum(list(filter(lambda x:  x > 0,PNL_REAL)))/len(list(filter(lambda x:  x > 0,PNL_REAL))) * 100 / _cash_ini,2))
        except:
            print('No winner')
        try:
            print('Average Loosers',round(sum(list(filter(lambda x:  x < 0,PNL_REAL)))/len(list(filter(lambda x:  x < 0,PNL_REAL))),2))
            print('% Average Loosers',round(sum(list(filter(lambda x:  x < 0,PNL_REAL)))/len(list(filter(lambda x:  x < 0,PNL_REAL))) / _cash_ini * 100,2))
        except:
            print('No looser')
        try:
            print('Average pnl',round(sum(PNL_REAL)/sum(TOTAL_OPEN),2))
            print('% Average pnl',round((sum(PNL_REAL)/len(set(PNL_REAL))) / _cash_ini * 100,2))
        except:
            print('No trade')
        
        print('Number of opened trades',sum(TOTAL_OPEN))
        print('Number of closed trades',sum(TOTAL_CLOSE))
        try:
            print('Max Exposure',max(EXPO_MAX),'x ',_size,'= ',max(EXPO_MAX)*_size,'$')
        except:
            print("Pas de trade => Pas d'exposure")
    candle_feedback = pd.DataFrame(index = price.index)
    candle_feedback['Symbol'] = x
    candle_feedback['Size'] = _size
    candle_feedback[_ticker] = EQUITY
    print('_bt_report :',_bt_report)
    
    if _bt_report == 1:
        joblib.dump(candle_feedback,'BT/'+_ticker+'_candle_feedback_oos.dag')

    elif _bt_report == 2:
        joblib.dump(candle_feedback,'BT/'+_ticker+'_candle_feedback_bt_bt.dag')
    
    elif _bt_report == 3:
        joblib.dump(candle_feedback,'BT/'+_ticker+'_candle_feedback_ai.dag')
    
    elif _bt_report == 4:
        joblib.dump(candle_feedback,'BT/'+_ticker+'_candle_feedback_custom.dag')
        
    return(TRACKER,df_resultats.T['Nbre Loosers'].sum())

# Determiner le body et les shadows des candles 
def klines(df):
    _condition1 = df.Close >= df.Open
    df['Color'] = np.where(_condition1,1,-1)
    _condition2 = df.Color = 1
    df['UpperShadow'] = np.where(_condition2,(df.High-df.Close),(df.High-df.Open))
    df['LowerShadow'] = np.where(_condition2,(df.Open-df.Low),(df.Close-df.Low))
    df['Body'] = abs(df.Close-df.Open)
    return (df)

# Split selon les dats en train / test / oos
def split_df(df,_start,_mid,_stop,_last):
    """[Split the dtaframe in train/test/oos and reduce train and test to dataframe with signal!=0 only]

    Args:
        df ([pandas]): [the datframe to split already featured]
        _start ([date]): [beginning of the df]
        _mid ([date]): [stop of the train and beginning of the test]
        _stop ([date]): [stop of the test and beginning of the oos]
        _last ([date]): [end of the oos]
    """    
    df = df.dropna()
    df['Date'] = pd.to_datetime(df.index)
    
    df_train = df[(df.Date>=_start)&(df.Date<=_mid)]
    df_test = df[(df.Date>_mid)&(df.Date<=_stop)]
    df_oos = df[(df.Date>_stop)&(df.Date <= _last)]
    df_train = df_train.drop(['Date'],axis=1)
    df_test = df_test.drop(['Date'],axis=1)
    df_oos = df_oos.drop(['Date'],axis=1)
    return(df_train, df_test,df_oos)

# Stochastic
def strategy(df):
    ##### CONDITIONS LONG
    _condition_1 = (df.slow_K5 < 20) & (df.slow_K5.shift(1) < df.slow_D5.shift(1)) & (df.slow_K5 > df.slow_D5)

    ##### CONDITIONS SHORT
    _condition_1_bar = (df.slow_K5 > 80) & (df.slow_K5.shift(1) > df.slow_D5.shift(1)) & (df.slow_K5 < df.slow_D5)

    ##### 1 condition
    df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))
    try:
        df = df.drop(['Symbol','Date','DateIndex','SB_Gamma'], axis=1)
    except:
        pass
    return(df.sort_index(axis=0)) 

# Gain Bedi'avad dans _window bougies
def strategy5(df,_window=40,_sl=0.001,_target=0.002):

    #df['RSI_2'] = TA.RSI(ohlc=df,int=2,str='Close')

    df['Window_High_Ask'] = df.HighAsk.iloc[::-1].rolling(_window).max().iloc[::-1] # Limite SL Short
    df['Window_High_Bid'] = df.HighBid.iloc[::-1].rolling(_window).max().iloc[::-1] # Limite Target Long
    df['Window_Low_Ask'] = df.LowAsk.iloc[::-1].rolling(_window).min().iloc[::-1] # Limite Target Short
    df['Window_Low_Bid'] = df.LowBid.iloc[::-1].rolling(_window).min().iloc[::-1] # Limite SL Long
    df['Window_sl_Short'] = df.CloseBid + (df.CloseBid * _sl) # Short pour SL
    df['Window_sl_Long'] = df.CloseAsk - (df.CloseAsk * _sl) # Long pour SL
    df['Window_tp_Short'] = df.CloseBid - (df.CloseBid * _target) # Short pour TP
    df['Window_tp_Long'] = df.CloseAsk + (df.CloseAsk * _target) # Long pour TP

    ##### CONDITIONS LONG
    _condition_1 = (df['Window_tp_Long'] <= df['Window_High_Bid']) & (df['Window_sl_Long'] <= df['Window_Low_Bid'])

    ##### CONDITIONS SHORT
    _condition_1_bar = (df['Window_tp_Short'] >= df['Window_Low_Ask']) & (df['Window_sl_Short'] >= df['Window_High_Ask'])

    ##### 1 condition
    df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))
    df = df.drop(['Symbol','Date','DateIndex','Window_High_Ask','Window_High_Bid','Window_Low_Ask','Window_Low_Bid','Window_sl_Short','Window_sl_Long','Window_tp_Short','Window_tp_Long'], axis=1)
    return(df.sort_index(axis=0))

def exotic_indicators(df):
    df['AOI'] = AwesomeOscillatorIndicator(df.High, df.Low, 135, 908, False).awesome_oscillator()
    df['KAMA_FAST'] = KAMAIndicator(df.Close, 10, 10, 30, False).kama()
    df['KAMA_SLOW'] = KAMAIndicator(df.Close, 10, 50, 30 , False).kama()
    df['ROC'] = ROCIndicator(df.Close, 324, False).roc()
    df['STOCH_RSI'] = StochRSIIndicator(df.Close, 378,81,81, False).stochrsi()
    df['STOCH_RSI3'] = StochRSIIndicator(df.Close, 378,81,81, False).stochrsi_k()
    df['STOCH_RSI10'] = StochRSIIndicator(df.Close, 378, 270, 270, False).stochrsi_k()
    df['TSI_SLOW'] = TSIIndicator(df.Close, 675,351, False).tsi()
    df['TSI_FAST'] = TSIIndicator(df.Close, 25,13, False).tsi()
    df['WRI_SLOW'] = WilliamsRIndicator(df.High, df.Low, df.Close, 378, False).williams_r()
    df['WRI_FAST'] = WilliamsRIndicator(df.High, df.Low, df.Close, 14, False).williams_r()
    df['DC_HB'] = DonchianChannel(df.High, df.Low, df.Close, 540,0, False).donchian_channel_hband()   # DC High Band
    df['DC_LB'] = DonchianChannel(df.High, df.Low, df.Close, 540,0, False).donchian_channel_lband()   # DC Low Band
    df['DC_MB'] = DonchianChannel(df.High, df.Low, df.Close, 540,0, False).donchian_channel_mband()   # DC Middle Band
    df['AI_D'] = AroonIndicator(df.Close, 675, False).aroon_down()
    df['AI_I'] = AroonIndicator(df.Close, 675, False).aroon_indicator()
    df['AI_U'] = AroonIndicator(df.Close, 675, False).aroon_up()
    return(df)
    
def rsi2(df):
    df['MYRSI_2'] = talib.RSI(df.Close,timeperiod=2)
    return(df)

def get_daily(df,_ticker):
    
    df['Date'] = df.index
    df['Date'] = df['Date'].dt.strftime(date_format='%Y-%m-%d')
    df1 = pd.DataFrame(index=df.Date.unique())
    df1['Lindex'] = list((df.groupby('Date').Date.first()))
    df1['Open'] = list((df.groupby('Date').Open.first()))
    df1['High'] = list((df.groupby('Date').High.max()))
    df1['Low'] = list((df.groupby('Date').Low.min()))
    df1['Close'] = list((df.groupby('Date').Close.last()))

    df1['Symbol'] = _ticker
    df1 = df1.sort_values('Lindex')
    reg = LinearRegression(n_jobs=-1)
    lr_window = 75

    for i in range(lr_window,len(df1)):
        df_X = df1[['Open','High','Low','Close']].iloc[i-lr_window:i,:].copy()
        df_Yopen = df1.loc[df1.index[i-lr_window+1:i+1],'Open'].copy()
        df_Yhigh = df1.loc[df1.index[i-lr_window+1:i+1],'High'].copy()
        df_Ylow = df1.loc[df1.index[i-lr_window+1:i+1],'Low'].copy()
        df_Yclose = df1.loc[df1.index[i-lr_window+1:i+1],'Close'].copy()
        _lr_open = reg.fit(df_X,df_Yopen.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'OpenS'] = _lr_open.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        _lr_high = reg.fit(df_X,df_Yhigh.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'HighS'] = _lr_high.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        _lr_low = reg.fit(df_X,df_Ylow.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'LowS'] = _lr_low.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        _lr_close = reg.fit(df_X,df_Yclose.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'CloseS'] = _lr_close.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]

    print('')
    print('Base daily, pour lr_window de',lr_window)
    print('Ecart moyen du Open',(df1.Open.shift(-1) - df1.OpenS).mean())
    print('Ecart moyen du High',(df1.High .shift(-1)- df1.HighS).mean())
    print('Ecart moyen du Low',(df1.Low.shift(-1) - df1.LowS).mean())
    print('Ecart moyen du Close',(df1.Close.shift(-1) - df1.CloseS).mean())
    print('')


    df1 = df1.sort_values('Lindex') ##########
    df1.set_index(pd.to_datetime(df1.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True) #####
    df1['Date'] = df1.Lindex
    df1 = df1.drop(['Lindex'],axis=1)
    df1 = df1.iloc[:-1,:]
    return(df1.sort_index(axis=0))

def get_weekly(df,_ticker):

    df1 = pd.DataFrame()
    df1['Lindex'] = list((df.groupby('Week').Date.first()))
    df1['Open'] = list((df.groupby('Week').Open.first()))
    df1['High'] = list((df.groupby('Week').High.max()))
    df1['Low'] = list((df.groupby('Week').Low.min()))
    df1['Close'] = list((df.groupby('Week').Close.last()))

    df1['Symbol'] = _ticker
    df1 = df1.sort_values('Lindex')
    reg = LinearRegression(n_jobs=-1)
    lr_window = 75

    for i in range(lr_window,len(df1)):
        df_X = df1[['Open','High','Low','Close']].iloc[i-lr_window:i,:].copy()
        df_Yopen = df1.loc[df1.index[i-lr_window+1:i+1],'Open'].copy()
        df_Yhigh = df1.loc[df1.index[i-lr_window+1:i+1],'High'].copy()
        df_Ylow = df1.loc[df1.index[i-lr_window+1:i+1],'Low'].copy()
        df_Yclose = df1.loc[df1.index[i-lr_window+1:i+1],'Close'].copy()
        _lr_open = reg.fit(df_X,df_Yopen.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'OpenS'] = _lr_open.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        _lr_high = reg.fit(df_X,df_Yhigh.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'HighS'] = _lr_high.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        _lr_low = reg.fit(df_X,df_Ylow.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'LowS'] = _lr_low.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        _lr_close = reg.fit(df_X,df_Yclose.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'CloseS'] = _lr_close.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]

    print('')
    print('Base weekly, pour lr_window de',lr_window)
    print('Ecart moyen du Open',(df1.Open.shift(-1) - df1.OpenS).mean())
    print('Ecart moyen du High',(df1.High .shift(-1)- df1.HighS).mean())
    print('Ecart moyen du Low',(df1.Low.shift(-1) - df1.LowS).mean())
    print('Ecart moyen du Close',(df1.Close.shift(-1) - df1.CloseS).mean())
    print('')
    
    df1 = df1.sort_values('Lindex')
    df1.set_index(pd.to_datetime(df1.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)
    df1['Symbol'] = _ticker
    df1['Date'] = df1.Lindex
    df1 = df1.drop(['Lindex'],axis=1) 
    #if df.index[-1].weekday() != 4:  
     #   df1 = df1.iloc[:-1,:]
    return(df1.sort_index(axis=0))

def import_rsi(df,df1,_suffix='hourly'):
    df1['RSI_2'] = talib.RSI(df1.Close,timeperiod =2)
    df1['RSI_14'] = talib.RSI(df1.Close,timeperiod =14)
    df['DateIndex'] = df.index
    df1['DateIndex'] = df1.index
    df1.loc[df1.index[-1] + pd.Timedelta('1 hours'),:] = 999
    df1[['RSI_2','RSI_14']] = df1[['RSI_2','RSI_14']].shift(1)
    df = df.join(df1[['RSI_2','RSI_14']],how='left',on='DateIndex',rsuffix=_suffix)
    df.rename(columns = {'RSI_2':'RSI_2_'+_suffix},inplace=True)
    df.rename(columns = {'RSI_14':'RSI_14_'+_suffix},inplace=True)
    try:
        df = df.drop(['Date'+_suffix],axis=1)
    except:
        pass        
    df['RSI_2_'+_suffix].fillna(method='ffill', inplace=True)
    df['RSI_14_'+_suffix].fillna(method='ffill', inplace=True)
    
    return(df.sort_index(axis=0))

def transform_H1(df1):
    
    reg = LinearRegression(n_jobs=-1)
    lr_window = 75

    for i in range(lr_window,len(df1)): # FAUX A CORRIGER SI BESOIN
        df_X = df1[['Open','High','Low','Close']].iloc[i-lr_window:i,:].copy()
        df_Yopen = df1.loc[df1.index[i-lr_window+1:i+1],'Open'].copy()
        df_Yhigh = df1.loc[df1.index[i-lr_window+1:i+1],'High'].copy()
        df_Ylow = df1.loc[df1.index[i-lr_window+1:i+1],'Low'].copy()
        df_Yclose = df1.loc[df1.index[i-lr_window+1:i+1],'Close'].copy()
        _lr_open = reg.fit(df_X,df_Yopen.to_numpy().reshape(-1, 1))
        _lr_high = reg.fit(df_X,df_Yhigh.to_numpy().reshape(-1, 1))
        _lr_low = reg.fit(df_X,df_Ylow.to_numpy().reshape(-1, 1))
        _lr_close = reg.fit(df_X,df_Yclose.to_numpy().reshape(-1, 1))
        df1.loc[df1.index[i],'OpenS'] = _lr_open.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        df1.loc[df1.index[i],'HighS'] = _lr_open.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        df1.loc[df1.index[i],'LowS'] = _lr_open.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]
        df1.loc[df1.index[i],'CloseS'] = _lr_open.predict(df1.loc[df1.index[i],['Open','High','Low','Close']].to_numpy().reshape(1,-1))[0][0]

    print('')
    print('Base hourly, pour lr_window de',lr_window)
    print('Ecart moyen du Open',(df1.Open.shift(-1) - df1.OpenS).mean())
    print('Ecart moyen du High',(df1.High .shift(-1)- df1.HighS).mean())
    print('Ecart moyen du Low',(df1.Low.shift(-1) - df1.LowS).mean())
    print('Ecart moyen du Close',(df1.Close.shift(-1) - df1.CloseS).mean())
    print('')

    #df1['OpenS'] = df1.Open.shift(1)
    #df1['HighS'] = df1.High.shift(1)
    #df1['LowS'] = df1.Low.shift(1)
    #df1['CloseS'] = df1.Close.shift(1)
    #df1.set_index(pd.to_datetime(df1.index,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)

    return(df1.sort_index(axis=0))

def timerange1D(df):
        df['Date'] = df.index
        df['Date'] = df['Date'].dt.strftime(date_format='%Y-%m-%d')
        return(df.sort_index(axis=0))

def timerange1W(df):
        df['WeekNo'] = pd.to_datetime(df.index)
        df['WeekNo'] = df['WeekNo'].dt.isocalendar().week.astype(str)
        df['Year'] = pd.to_datetime(df.index)
        df['Year'] = df['Year'].dt.year.astype(str)
        df['Date'] = pd.to_datetime(df.index)
        df['list']=df[['Year','WeekNo']].values.tolist()
        df['Week']=df['list'].apply('_'.join)
        #df.sort_values('Date')
        return(df.sort_index(axis=0))

def Wilder(data, window):
    start = np.where(~np.isnan(data))[0][0] # Positionne après les nan
    Wilder = np.array([np.nan]*len(data)) # Replace les nan en début de liste pour ne pas changer la longueur
    Wilder[start+window-1] = data[start:(start+window)].mean() #Simple Moving Average pour la window window
    for i in range(start+window,len(data)):
        Wilder[i] = ((Wilder[i-1]*(window-1) + data[i])/window) #Wilder Smoothing
    return(Wilder)

def ema(df, _window):
    df['EMA_'+str(_window)] = df.Close.ewm(span=_window,adjust=False).mean()
    return(df.sort_index(axis=0))

def sma(df,_window=200):
    df['SMA_'+str(_window)] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = _window).mean())
    return(df.sort_index(axis=0))

def slowstochastic(df,_window=5,_per=3):
    df['Lowest_'+str(_window)] = df['Low'].transform(lambda x: x.rolling(window = _window).min())
    df['Highest_'+str(_window)] = df['High'].transform(lambda x: x.rolling(window = _window).max())
    df['slow_K'+str(_window)] = (((df['Close'] - df['Lowest_'+str(_window)])/(df['Highest_'+str(_window)] - df['Lowest_'+str(_window)]))*100).rolling(window = _per).mean()
    df['slow_D'+str(_window)] = df['slow_K'+str(_window)].rolling(window = _per).mean()
    df = df.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)
    return(df.sort_index(axis=0))

def rsi_home(df,_window=5):
    print(col.Fore.MAGENTA+'\nCalcul RSI'+col.Style.RESET_ALL)
    
    ##### Pour chaque Symbol, Calcule la différence du close de la cellule précédente à la cellule actuelle
    df['Diff'] = df['Close'].diff()
    ##### Ne garde que les valeurs positives et met 0 sinon
    df['Up'] = df['Diff']
    df.loc[(df['Up']<0), 'Up'] = 0
    ##### Pour chaque Symbol, Calcule la différence du close de la cellule précédente à la cellule actuelle
    df['Down'] = df['Diff']
    ##### Ne garde que les valeurs négatives et met 0 sinon. Passe ensuite les valeurs négatives en valeur absolue
    df.loc[(df['Down']>0), 'Down'] = 0 
    df['Down'] = abs(df['Down'])

    ##### Calcule sur les fast & slow les moyennes des UP est DOWN créés
    df['avg_up'+str(_window)] = df['Up'].rolling(window=_window).mean()
    df['avg_down'+str(_window)] = df['Down'].rolling(window=_window).mean()

    ##### Pour les fast & slow, calcule le ratio de (moyenne UP / moyenne DOWN)
    df['RS_'+str(_window)] = df['avg_up'+str(_window)] / df['avg_down'+str(_window)]

    ##### Le RSI fast & slow peut alors être calculé
    ##### 100 - (100/(1 + RS))
    df['RSI_'+str(_window)] = 100 - (100/(1+df['RS_'+str(_window)]))

    df = df.drop(['Diff','Up','Down','avg_up'+str(_window),'avg_down'+str(_window),'RS_'+str(_window)],axis=1)

    return(df.sort_index(axis=0))

def rsi(df,_window=14):
    df['RSI_'+str(_window)] = talib.RSI(df.Close,_window)
    return(df)

def stochrsi(df,_window=14):
    df['RSI_'+str(_window)] = talib.RSI(df.Close,_window)
    df['RSI_Stoch'] = ((df['RSI_'+str(_window)] - df['RSI_'+str(_window)].rolling(_window).min())/(df['RSI_'+str(_window)].rolling(_window).max() - df['RSI_'+str(_window)].rolling(_window).min())) * 100
    return(df.sort_index(axis=0))

def bollinger(df,_slow=15):
    df['MA'+str(_slow)] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=_slow).mean())
    df['SD'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=_slow).std())
    df['UpperBand'] = df['MA'+str(_slow)] + 2*df['SD']
    df['LowerBand'] = df['MA'+str(_slow)] - 2*df['SD']
    df = df.drop(['MA'+str(_slow),'SD'],axis=1)
    return(df.sort_index(axis=0))

def onlosma(df,_window=8):
    df['ONLOSMA_'+str(_window)] = df.Low.rolling(_window).mean()
    return(df.sort_index(axis=0))

def onhisma(df,_window=8):
    df['ONHISMA_'+str(_window)] = df.High.rolling(_window).mean()
    return(df.sort_index(axis=0))

def atr(df,_ticker,_window=14):
    df['prev_close'] = df['Close'].shift(1)

    df['TR'] = np.maximum((df['High'] - df['Low']), 
                        np.maximum(abs(df['High'] - df['prev_close']), 
                        abs(df['prev_close'] - df['Low'])))

    df.loc[df.Symbol==_ticker,'ATR_'+str(_window)] = Wilder(df['TR'], _window)
    return(df.sort_index(axis=0))       

def pivot(df):
    df['PP'] = (df.High + df.Low + df.Close) / 3
    df['S38'] = df.PP - (0.382 * (df.High - df.Low))
    df['S62'] = df.PP - (0.618 * (df.High - df.Low))
    df['S78'] = df.PP - (0.78 * (df.High - df.Low))
    df['S100'] = df.PP - (1 * (df.High - df.Low))
    df['S138'] = df.PP - (1.382 * (df.High - df.Low))
    df['S162'] = df.PP - (1.618 * (df.High - df.Low))
    df['S178'] = df.PP - (1.78 * (df.High - df.Low))
    df['S200'] = df.PP - (2 * (df.High - df.Low))
    df['R38'] = df.PP + (0.382 * (df.High - df.Low))
    df['R62'] = df.PP + (0.618 * (df.High - df.Low))
    df['R78'] = df.PP + (0.78 * (df.High - df.Low))
    df['R100'] = df.PP + (1 * (df.High - df.Low))
    df['R138'] = df.PP + (1.382 * (df.High - df.Low))
    df['R162'] = df.PP + (1.618 * (df.High - df.Low))
    df['R178'] = df.PP + (1.78 * (df.High - df.Low))
    df['R200'] = df.PP + (2 * (df.High - df.Low))
    return(df.sort_index(axis=0))

def pivotimportdf(df,df1):
    """[Import from df1 to df all the pivots]

    Args:
        df ([pandas]): [Where you've got to import pivot : ex : intraday]
        df1 ([pandas]): [from where you import pivo : ex : weekly]
    """    
    df1['Date'] = pd.to_datetime(df1.Date)
    df['Date'] = pd.to_datetime(df.Date)
    df1.loc[df1.index[-1] + pd.Timedelta('7 days'),:] = 999
    df1[['PP' ,'S38', 'S62', 'S78' ,'S100'  ,'S138'  ,'S162' ,'S178','S200'  ,'R38'  ,'R62','R78' ,'R100','R178' ,'R138' ,'R162' ,'R200' ]] = \
        df1[['PP' ,'S38', 'S62', 'S78' ,'S100'  ,'S138'  ,'S162' ,'S178','S200'  ,'R38'  ,'R62','R78' ,'R100','R178' ,'R138' ,'R162' ,'R200' ]].shift(1)
    
    df = df.join(df1[['PP','S38','S62','S78','S100','S138','S162','S178','S200','R38','R62','R78','R100','R138','R162','R178','R200','Date']],how='left',on='Date',rsuffix='_2drop')
    df = df.drop(['Date_2drop'],axis=1)
    df.PP.fillna(method='ffill', inplace=True)
    df.S38.fillna(method='ffill', inplace=True)
    df.S62.fillna(method='ffill', inplace=True)
    df.S78.fillna(method='ffill', inplace=True)
    df.S100.fillna(method='ffill', inplace=True)
    df.S138.fillna(method='ffill', inplace=True)
    df.S162.fillna(method='ffill', inplace=True)
    df.S178.fillna(method='ffill', inplace=True)
    df.S200.fillna(method='ffill', inplace=True)
    df.R38.fillna(method='ffill', inplace=True)
    df.R62.fillna(method='ffill', inplace=True)
    df.R78.fillna(method='ffill', inplace=True)
    df.R100.fillna(method='ffill', inplace=True)
    df.R138.fillna(method='ffill', inplace=True)
    df.R162.fillna(method='ffill', inplace=True)
    df.R178.fillna(method='ffill', inplace=True)
    df.R200.fillna(method='ffill', inplace=True)
    return(df.sort_index(axis=0))

def adr(df,_window):
    df['ADR'] = (df.High - df.Low).rolling(_window).mean()#.shift(1)
    df['ADR_High'] = df['Low'] + df['ADR']
    df['ADR_Low'] = df['High'] - df['ADR']
    df = df.drop(['list','Week','WeekNo','Year'],axis=1)
    return(df.sort_index(axis=0))

def rvi(df,_window):
    df['Std'] = df.Close.rolling(window=_window).std()
    df['Positive'] = np.where((df.Std > df.Std.shift(1)),df.Std,0)
    df['Negative'] = np.where((df.Std < df.Std.shift(1)),df.Std,0)
    df['PoMA'] = Wilder(df['Positive'],_window)
    df['NeMA'] = Wilder(df['Negative'],_window)
    df['RVI'] = (100 * df['PoMA']) / (df['PoMA'] + df['NeMA'])
    df = df.drop(['Std','Positive','Negative','PoMA','NeMA'],axis=1)
    return(df.sort_index(axis=0))

def getadr(df,df1):
    """[Get adr from df to df1]

    Args:
        df ([pandas]): [ex : daily]
        df1 ([pandas]): [ex : df intraday]
    """    
    _suffix='_2Drop'
    if df.index[-1].weekday() != 4:
        df.loc[df.index[-1] + pd.Timedelta('1 days'),:] = 999
    else:
        df.loc[df.index[-1] + pd.Timedelta('3 days'),:] = 999
        
    df['Date'] = pd.to_datetime(df.index)
    df1['Date'] = pd.to_datetime(df1.Date)


    df['ADR_High'] = df['ADR_High'].shift(1)
    df['ADR_Low'] = df['ADR_Low'].shift(1)
    #df['High'] = df['High'].shift(1)
    #df['Low'] = df['Low'].shift(1)
    #df1 = df1.join(df[['ADR','High','Low']],how='left',on='Date',rsuffix=_suffix)
    #df1 = df1.rename(columns={'High'+_suffix: "DayHigh", 'Low'+_suffix: "DayLow"})
    df1 = df1.join(df[['ADR_High']],how='left',on='Date',rsuffix=_suffix)
    df1 = df1.join(df[['ADR_Low']],how='left',on='Date',rsuffix=_suffix)
    try:
        df1 = df1.drop(['Date'+_suffix],axis=1)
    except:
        pass
    df1['ADR_High'].fillna(method='ffill', inplace=True)
    df1['ADR_Low'].fillna(method='ffill', inplace=True)
   
    return(df1.sort_index(axis=0))

def adrhnl_original(df):
    global _flagh, _flagl , val
    _flagh = 0
    _flagl = 0
    val = 0

    def fh(row):
        global _flagh, val
    
        if row['High'] < row['DayHigh'] and row['High'] <= row['HighShift'] and _flagh == 0 and row['Date'] == row['DateShiftMinus'] and row['Date'] == row['DateShiftPlus']:
            val = row['HighShift'] # np.nan
            _flagh = 0
        
        elif row['High'] < row['DayHigh'] and row['High'] > row['HighShift'] and _flagh == 0 and row['Date'] == row['DateShiftMinus'] and row['Date'] == row['DateShiftPlus']:
            val = row['High'] # 
            _flagh = 0
                
        return(val)

    def fl(row):
        global _flagl, val
        
        if row['Low'] > row['DayLow'] and row['Low'] >= row['LowShift'] and _flagl == 0 and row['Date'] == row['DateShiftMinus'] and row['Date'] == row['DateShiftPlus']:
            _flagl = 0
            val =  row['LowShift'] # np.nan
        
        elif row['Low'] > row['DayLow'] and row['Low'] < row['LowShift'] and _flagl == 0 and row['Date'] == row['DateShiftMinus'] and row['Date'] == row['DateShiftPlus']:
            val = row['Low']
            _flagl = 0

        return(val)

    
    df['DateShiftMinus'] = df.Date.shift(1)
    df['DateShiftPlus'] = df.Date.shift(-1)

    df['HighShift'] = df.High.shift(1)
    df['LowShift'] = df.Low.shift(1)

    #df['DayHigh'] = df['High'].groupby(df['Date']).max()
    #df['DayHigh'].fillna(method='ffill', inplace=True)
    #df['DayLow'] = df['Low'].groupby(df['Date']).min()
    #df['DayLow'].fillna(method='ffill', inplace=True)


    df['HighSlope'] = df.apply(fh,axis=1)
    df['LowSlope'] = df.apply(fl,axis=1)

    df['ADR_High'] = df.LowSlope + df.ADR
    df['ADR_Low'] = df.HighSlope - df.ADR

    df = df.drop(['DateShiftMinus','DateShiftPlus','HighShift','LowShift','HighSlope','LowSlope','DayHigh','DayLow'],axis=1)
    return(df.sort_index(axis=0))

def adrhnl(df):
    df['ADR_High'] = df.Low + df.ADR
    df['ADR_Low'] = df.High - df.ADR
    return(df.sort_index(axis=0))

def sbgamma(df):      
    _op1 = (df.Close - df.Open)/(df.Close.shift(1) - df.Open.shift(1))
    _op2 = (df.Close - df.Open)/(df.CloseAsk.shift(1) - df.OpenAsk.shift(1))
    _op3 = (df.Close - df.Open)/(df.CloseBid.shift(1) - df.OpenBid.shift(1))
    _op4 = (df.Close - df.Open)/(df.CloseBid.shift(1) - df.OpenAsk.shift(1))
    _op5 = (df.Close - df.Open)/(df.CloseAsk.shift(1) - df.OpenBid.shift(1))

    _condition1 = df.Close.shift(1) != df.Open.shift(1)
    _condition2 = df.CloseAsk.shift(1) != df.OpenAsk.shift(1)
    _condition3 = df.CloseBid.shift(1) != df.OpenBid.shift(1)
    _condition4 = df.CloseBid.shift(1) != df.OpenAsk.shift(1)
    _condition5 = df.CloseAsk.shift(1) != df.OpenBid.shift(1)

    df['SB_Gamma'] = np.where(_condition1,_op1,np.where(_condition2,_op2,np.where(_condition3,_op3,np.where(_condition4,_op4,np.where(_condition5,_op5,1.93E13)))))
    return(df.sort_index(axis=0))

def importohlc(df,df1,_suffix):
    df['Date'] = pd.to_datetime(df.Date)
    df1['Date'] = pd.to_datetime(df1.index)
    
    if _suffix=='daily':
        if df1.index[-1].weekday() != 4:
            df1.loc[df1.index[-1] + pd.Timedelta('1 days'),:] = 999
        else:
            df1.loc[df1.index[-1] + pd.Timedelta('3 days'),:] = 999
    
    if _suffix=='weekly':
        if df1.index[-1].weekday() != 0:
            df1.loc[df1.index[-1] + pd.Timedelta('7 days'),:] = 999
        
    df1[['OpenS','HighS','LowS','CloseS']] = df1[['OpenS','HighS','LowS','CloseS']].shift(1)
    df = df.join(df1[['OpenS','HighS','LowS','CloseS']],how='left',on='Date',rsuffix=_suffix)
    df.rename(columns = {'OpenS':'OpenShift'+_suffix,'HighS':'HighShift'+_suffix,'LowS':'LowShift'+_suffix,'CloseS':'CloseShift'+_suffix},inplace=True)
    
    try:
        df = df.drop(['Date'+_suffix],axis=1)
    except:
        pass        
    df['OpenShift'+_suffix].fillna(method='ffill', inplace=True)
    df['HighShift'+_suffix].fillna(method='ffill', inplace=True)
    df['LowShift'+_suffix].fillna(method='ffill', inplace=True)
    df['CloseShift'+_suffix].fillna(method='ffill', inplace=True)
    return(df.sort_index(axis=0))

def init_base(x,_period,_period2):
    _ticker = x.replace('/','')
    """[Première initialisation de la base Live à partir de la base HDD et vérifications d'usage]
    """    
    _t1 = dt.datetime.now()
    print('Début des opérations horodatée à',col.Fore.YELLOW,dt.datetime.now(),col.Style.RESET_ALL)
    
    print('\nINITIALISATION DE LA BASE\n')

    print('Ticker :',col.Fore.YELLOW,x,col.Style.RESET_ALL)

    df,df_H1 = load_hdd()
    
    df = drop_we(df)
    is_we(df)
    
    df_H1 = drop_we(df_H1)
    is_we(df_H1)

    df = make_mid(df)
    
    df_H1 = make_mid(df_H1)

    df = reduce_df(df)

    df_H1 = reduce_df(df_H1)

    #df, df_H1, df_D1, df_W1 =  make_indicators(df, df_H1)

    engine.say("The job is done")
    engine.runAndWait()

    print('Sauvegarde des Bases')
    joblib.dump(df_H1,'BASES/'+_ticker+'_'+_period2)
    joblib.dump(df,'BASES/'+_ticker+'_'+_period)
    #joblib.dump(df_D1,'BASES/'+_ticker+'_D1')
    #joblib.dump(df_W1,'BASES/'+_ticker+'_W1')
    print('Bases sauvegardées')

    engine.say("All the bases are saved")
    engine.runAndWait()

    print('\ndf :',df,'\n')
    print('\ndf_H1 :',df_H1,'\n')
    #print('\ndf_D1 :',df_D1,'\n')
    #print('\ndf_W1 :',df_W1,'\n') 

    #print('\nAnalyse des nan dans df :')
    #check_nan(df)

    #print('\nAnalyse des inf :')
    #check_inf(df)

    #print('\n Analyse des bougies manquantes :')
    #missing_candle_hdd(df)

    _t2 = dt.datetime.now()
    print('Fin des opérations horodatée à',col.Fore.YELLOW,dt.datetime.now(),col.Style.RESET_ALL)
    print('Executé en :',(_t2 - _t1))
    return(df,df_H1)

def is_we(dataframe_to_check):
    IDX = dataframe_to_check.index.to_list()
    c=0
    for day in tqdm(IDX):
        if day.weekday() == 5 or day.weekday() == 6:
            c += 1
    print('Nombre de samedi et dimanches présents :',c)

def load_hdd(_ticker,_period,_period2):

    engine.say("Loading raw data")
    engine.runAndWait()
    
    df = pd.read_csv('HDD/'+_ticker+'_'+_period+'_BidAndAsk.csv')

    ##### Ajout de la colonne Symbol pour identifier le ticker
    df['Symbol'] = _ticker

    ##### On fixe la date en index sous forme de Timestamp
    df['Lindex'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index(pd.to_datetime(df.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)

    ###### On drop les colonnes inutiles
    df = df.drop(['Date','Lindex','Time','Total Ticks'],axis=1)


    df_H1 = pd.read_csv('HDD/'+_ticker+'_'+_period2+'_BidAndAsk.csv')

    ##### Ajout de la colonne Symbol pour identifier le ticker
    df_H1['Symbol'] = _ticker

    ##### On fixe la date en index sous forme de Timestamp
    df_H1['Lindex'] = pd.to_datetime(df_H1['Date'] + ' ' + df_H1['Time'])
    df_H1.set_index(pd.to_datetime(df_H1.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)

    ###### On drop les colonnes inutiles
    df_H1 = df_H1.drop(['Date','Lindex','Time','Total Ticks'],axis=1)

    engine.say("Raw data are loaded")
    engine.runAndWait()    
    return(df,df_H1)

def drop_we(df):
    df['WE'] = np.where(((df.index.weekday == 5) | (df.index.weekday == 6)),None,df.index.weekday)
    df = df.dropna()
    df = df.drop(['WE'],axis=1)
    return(df)

def make_mid(df,_ticker):
    df['Open'] = (df.OpenAsk + df.OpenBid)/2
    df['High'] = (df.HighAsk + df.HighBid)/2
    df['Low'] = (df.LowAsk + df.LowBid)/2
    df['Close'] = (df.CloseAsk + df.CloseBid)/2
    df['Symbol'] = _ticker
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'].dt.strftime(date_format='%Y-%m-%d'))
    df = drop_we(df)
    return(df)

def make_indicators(df, df_H1):
    # Make daily & weekly bases

    df = timerange1D(df)
    df_H1 = timerange1D(df_H1)
    #df_H1 = transform_H1(df_H1)
    df_D1 = get_daily(df_H1)

    df_D1 = timerange1W(df_D1)
    df_W1 = get_weekly(df_D1)

    # Calculate the indicators
    df_D1 = adr(df_D1,_window=14)
    df = getadr(df_D1,df)
    #df = adrhnl(df)
    df = sma(df=df,_window=200)
    df = bollinger(df,_slow=20)
    df = slowstochastic(df)
    #df = stochrsi(df,_window = 14)
    df = ema(df,21)
    df = ema(df,8)

    df_W1 = pivot(df_W1)
    df = pivotimportdf(df,df_W1)
    df = atr(df,14)
    df = rvi(df,_window=14)
    df = rsi2(df)
    df = sbgamma(df)
    df = onhisma(df,_window=5)
    df = onlosma(df,_window=5)
    df = onhisma(df,_window=21)
    df = onlosma(df,_window=21)
    df = onhisma(df,_window=34)
    df = onlosma(df,_window=34)
    df = importohlc(df,df_W1,_suffix='_weekly')
    df = importohlc(df,df_D1,_suffix='_daily')
    #df = importohlc(df,df_D1,_suffix='_hourly')
    df = import_rsi(df,df_H1)

    return(df, df_H1, df_D1, df_W1)

def reduce_df(df):
    df = df[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
    return(df)


if __name__ == '__main__':
    pass





