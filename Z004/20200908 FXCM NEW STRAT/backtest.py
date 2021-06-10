###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
##################################      B A C K T E S T      O N       F X C M       ######################################
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
import colorama as col
import pandas as pd
import time
import os
import datetime as dt
import numpy as np
import shelve
import talib
import statistics
import configparser
from scraptickers import scrap_tickers
import pyttsx3

engine = pyttsx3.init()

print('Librairies imported\n')
engine.say("Processing the backtest")
engine.runAndWait()




############################
######### DEF BT ###########
############################

# globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period1+'.csv')

print('Variables initialisées')

def bt(x,_cash):
    df_resultats = pd.DataFrame(index=['TimeFrame 1','TimeFrame2','Equity','Nbre Winners','Nbre Loosers','Average lenght of trade','Cumul pnl'])
    price = pd.read_csv('Base_Signals/'+x.replace('/','')+'m5'+'.csv')
    price = price.iloc[-int(len(price) * 0.2):,:]
    price['Close'] = (price.CloseBid + price.CloseAsk)/2
    price['High'] = (price.HighBid + price.HighAsk)/2
    price['Low'] = (price.LowBid + price.LowAsk)/2
    price['Open'] = (price.OpenBid + price.OpenAsk)/2

    print(price.head(1))

    _position = 0
    _equity = 0
    _nbtransactions = 0
    backtest_graph = pd.DataFrame()
    EQUITY = [_cash]
    CASH = [_cash]
    _winner = 0
    _looser = 0
    _index_entry = 0
    TRADE_DURATION = []
    _average_duration = 0
    _size = 50000

    PRICE_BUY = []
    PRICE_SELL = []
   
    _total = 0

    _filtre_multipoz = 0.2 # En %
    _filtre_multipoz = _filtre_multipoz/100
    _print_chart = 'no'
    _print_patterns = 'no' 

    _open_buy = 0
    _open_sell = 0
    _minipoz = 0

    MM21 = talib.EMA(price.Close, timeperiod=21)
    MM50 = talib.EMA(price.Close, timeperiod=50)
    MM200 = talib.EMA(price.Close, timeperiod=200)
    rsi = talib.RSI(price.Close, timeperiod=14)
    rsi_high = 70
    rsi_low = 30
    upperband, middleband, lowerband = talib.BBANDS(price.Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    fastk, fastd = talib.STOCHRSI(price.Close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)#STOCHASTICRSI
    slowk, slowd = talib.STOCH(price.High, price.Low, price.Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)#STOCHASTIC

    if _position == 0 :
        if slowk[i-1] > 80 and slowk[i-1] > slowd[i-1] and slowk[i] < slowd[i] and df1.High[i-1] > df1.High[i-2] and df1.High[i] < df1.High[i-2]:

            _pnl = 0
            _open_sell += 1
            _equity = _cash + _pnl
            EQUITY.append(_equity)
            CASH.append(_cash)
            _index_entry = i
            _position = -1
            _nbtransactions += 1
            price_sell = price.Close_Bid.iloc[i]
            PRICE_SELL.append(price_sell)
            _price_sell_mean = round(sum(PRICE_SELL)/len(PRICE_SELL),5)
            print('Position 1 sold at', price_sell,'(verification liste',PRICE_SELL[-1],')')
            print('Nouvelle moyenne du price_sell',_price_sell_mean)

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

        elif slowk[i-1] < 20 and slowk[i-1] < slowd[i-1] and slowk[i] > slowd[i]  and df1.Low[i-1] < df1.Low[i-2] and df1.Low[i] > df1.Low[i-2]:
                
            _pnl = 0
            _open_buy += 1
            _equity = _cash + _pnl
            EQUITY.append(_equity)
            CASH.append(_cash)
            _position = 1
            _index_entry = i
            _nbtransactions += 1
            price_buy = price.Close_Ask.iloc[i]
            PRICE_BUY.append(price_buy)
            _price_buy_mean = round(sum(PRICE_BUY)/len(PRICE_BUY),5)
            print('Position 1 bought at', price_buy,'(verification liste',PRICE_BUY[-1],')')
            print('Nouvelle moyenne du price_buy',_price_buy_mean)

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

        else :
            _pnl = 0
            _equity = _cash + _pnl
            EQUITY.append(_equity)
            CASH.append(_cash)
            
    if _position == 1:

        if slowk[i-1] < 20 and slowk[i-1] < slowd[i-1] and slowk[i] > slowd[i]  and df1.Low[i-1] < df1.Low[i-2] and df1.Low[i] > df1.Low[i-2]:

            _pnl = (price.Close_Bid.iloc[i] - _price_buy_mean) * _size * _open_buy
            _open_buy += 1
            _equity = _cash + _pnl
            EQUITY.append(_equity)
            CASH.append(_cash)
            _position = 1
            _index_entry = i
            _nbtransactions += 1
            price_buy = price.Close_Ask.iloc[i]
            PRICE_BUY.append(price_buy)
            _price_buy_mean = round(sum(PRICE_BUY)/len(PRICE_BUY),5)
            print('Position',_open_buy,'bought at ', price_buy,'(verification liste',PRICE_BUY[-1],')')
            print('Nouvelle moyenne du price_buy',_price_buy_mean)

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
        
        elif (float(price.Close_Bid.iloc[i]) - float(_price_buy_mean))/float(_price_buy_mean) >= _target :

            _position = 0
            _pnl = (price.Close_Bid.iloc[i] - _price_buy_mean) * _size * _open_buy
            _pnl,_flag = convert(_pnl,_rate,_flag)
            _total += _pnl
            _cash += _pnl
            _equity = _cash
            EQUITY.append(_equity)
            CASH.append(_cash)
            _winner += _open_buy

            TRADE_DURATION.append(i - _index_entry)
            print(i)
            print(_open_buy,'positions TP closed at',price.Close_Bid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
            print('nombre de candles en position :',i - _index_entry)
            print('Equity :', _equity)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(0)
            CLOSE_POZ.append(1)
            RATE_OPEN_POZ.append(0)
            RATE_CLOSE_POZ.append(price.Close_Bid.iloc[i])
            PNL_LAT.append(0)
            PNL_REAL.append(_pnl)
            TOTAL_PNL_LAT.append(0)
            TOTAL_PNL_REAL.append(_pnl)
            TOTAL_CLOSE.append(_open_buy)
            PRICE_BUY = [] 
            _open_buy = 0

        elif (float(price.Close_Bid.iloc[i]) - float(_price_buy_mean))/float(_price_buy_mean) <= -_sl :

            _position = 0
            _pnl = (price.Close_Bid.iloc[i] - _price_buy_mean) * _size * _open_buy
            _pnl,_flag = convert(_pnl,_rate,_flag)
            _total += _pnl
            _cash += _pnl
            _equity = _cash
            EQUITY.append(_equity)
            CASH.append(_cash)
            _looser += _open_buy

            TRADE_DURATION.append(i - _index_entry)
            print(i)
            print(_open_buy,'positions SL closed at',price.Close_Bid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
            print('nombre de candles en position :',i - _index_entry)
            print('Equity :', _equity)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(0)
            CLOSE_POZ.append(1)
            RATE_OPEN_POZ.append(0)
            RATE_CLOSE_POZ.append(price.Close_Bid.iloc[i])
            PNL_LAT.append(0)
            PNL_REAL.append(_pnl)
            TOTAL_CLOSE.append(_open_buy) 
            TOTAL_PNL_LAT.append(0)
            TOTAL_PNL_REAL.append(_pnl)
            PRICE_BUY = []
            _open_buy = 0

        else:
                
                _pnl = (price.Close_Bid.iloc[i] - _price_buy_mean) * _size * _open_buy
                _pnl,_flag = convert(_pnl,_rate,_flag)
                _equity = _cash + _pnl
                EQUITY.append(_equity)
                CASH.append(_cash)

                DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
                CONTRACT.append(x)
                OPEN_POZ.append(0)
                CLOSE_POZ.append(0)
                RATE_OPEN_POZ.append(0)
                RATE_CLOSE_POZ.append(0) ############### 
                PNL_LAT.append(_pnl)
                PNL_REAL.append(0)
                TOTAL_PNL_LAT.append(_pnl)
                TOTAL_PNL_REAL.append(0) 


    elif _position == -1:

        if slowk[i-1] > 80 and slowk[i-1] > slowd[i-1] and slowk[i] < slowd[i] and df1.High[i-1] > df1.High[i-2] and df1.High[i] < df1.High[i-2]:

            _pnl = - (price.Close_Ask.iloc[i] - _price_sell_mean) * _size * _open_sell
            _open_sell += 1
            _equity = _cash + _pnl
            EQUITY.append(_equity)
            CASH.append(_cash)
            _position = -1
            _index_entry = i
            _nbtransactions += 1
            price_sell = price.Close_Bid.iloc[i]
            PRICE_SELL.append(price_sell)
            _price_sell_mean = round(sum(PRICE_SELL)/len(PRICE_SELL),5)
            print('Position 1 sold at', price_sell,'(verification liste',PRICE_SELL[-1],')')
            print('Nouvelle moyenne du price_sell',_price_sell_mean)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(1)
            CLOSE_POZ.append(0)
            RATE_OPEN_POZ.append(price_sell)
            RATE_CLOSE_POZ.append(0)
            PNL_LAT.append(_pnl)
            PNL_REAL.append(0)
            TOTAL_OPEN.append(1) 
            TOTAL_PNL_LAT.append(_pnl)
            TOTAL_PNL_REAL.append(0) 



        elif (float(price.Close_Ask.iloc[i]) - float(_price_sell_mean))/float(_price_sell_mean) <= -_target :

            _position = 0
            _pnl = - (price.Close_Ask.iloc[i] - _price_sell_mean) * _size * _open_sell
            _pnl,_flag = convert(_pnl,_rate,_flag)
            _total += _pnl
            _cash += _pnl
            _equity = _cash
            EQUITY.append(_equity)
            CASH.append(_cash)
            _winner += _open_sell
            TRADE_DURATION.append(i - _index_entry)
            print(_open_sell,'positions TP closed at',price.Close_Ask.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
            print('nombre de candles en position :',i - _index_entry)
            print('Equity :', _equity)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(0)
            CLOSE_POZ.append(-1)
            RATE_OPEN_POZ.append(0)
            RATE_CLOSE_POZ.append(price.Close_Ask.iloc[i])
            PNL_LAT.append(0)
            PNL_REAL.append(_pnl)
            TOTAL_PNL_LAT.append(0)
            TOTAL_PNL_REAL.append(_pnl)
            TOTAL_CLOSE.append(_open_sell)
            PRICE_SELL = [] 
            _open_sell = 0

        elif (float(price.Close_Ask.iloc[i]) - float(_price_sell_mean))/float(_price_sell_mean) >= _sl :
            _position = 0
            _pnl = - (price.Close_Ask.iloc[i] - _price_sell_mean) * _size * _open_sell
            _pnl,_flag = convert(_pnl,_rate,_flag)
            _total += _pnl
            _cash += _pnl
            _equity = _cash
            EQUITY.append(_equity)
            CASH.append(_cash)
            _looser += _open_sell
            TRADE_DURATION.append(i - _index_entry)
            print(i)
            print(_open_sell,'position SL closed at',price.Close_Ask.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
            print('nombre de candles en position :',i - _index_entry)
            print('Equity :', _equity)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(0)
            CLOSE_POZ.append(-1)
            RATE_OPEN_POZ.append(0)
            RATE_CLOSE_POZ.append(price.Close_Ask.iloc[i])
            PNL_LAT.append(0)
            PNL_REAL.append(_pnl)
            TOTAL_PNL_LAT.append(0)
            TOTAL_PNL_REAL.append(_pnl)
            TOTAL_CLOSE.append(_open_sell)
            PRICE_SELL = []
            _open_sell = 0

        else:
            
            _pnl = - (price.Close_Ask.iloc[i] - _price_sell_mean) * _size * _open_sell
            _pnl,_flag = convert(_pnl,_rate,_flag)
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
                                          
    elif i == (len(price)-1) and _position != 0 :

        if _position == -1:
            _position = 0
            _pnl = - (price.Close_Ask.iloc[i] - _price_sell_mean) * _size * _open_sell
            _pnl,_flag = convert(_pnl,_rate,_flag)
            _total += _pnl
            _cash += _pnl
            _equity = _cash
            EQUITY.append(_equity)
            CASH.append(_cash)
            _looser += _open_sell
            TRADE_DURATION.append(i - _index_entry)
            print(col.Fore.CYAN,"Cloture des positions en l'air",col.Style.RESET_ALL)
            print(_open_sell,'position closed at',price.Close_Ask.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
            print('nombre de candles en position :',i - _index_entry)
            print('Equity :', _equity)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(0)
            CLOSE_POZ.append(-1)
            RATE_OPEN_POZ.append(0)
            RATE_CLOSE_POZ.append(price.Close_Ask.iloc[i])
            PNL_LAT.append(0)
            PNL_REAL.append(_pnl)
            TOTAL_PNL_LAT.append(0)
            TOTAL_PNL_REAL.append(_pnl)
            TOTAL_CLOSE.append(_open_sell)
            PRICE_SELL = []
            _open_sell = 0

        if _position == 1:

            _position = 0
            _pnl = (price.Close_Bid.iloc[i] - _price_buy_mean) * _size * _open_buy
            _pnl,_flag = convert(_pnl,_rate,_flag)
            _total += _pnl
            _cash += _pnl
            _equity = _cash
            EQUITY.append(_equity)
            CASH.append(_cash)
            _looser += _open_buy

            TRADE_DURATION.append(i - _index_entry)
            print(col.Fore.CYAN,"Cloture des positions en l'air",col.Style.RESET_ALL)
            print(_open_buy,'positions closed at',price.Close_Bid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
            print('nombre de candles en position :',i - _index_entry)
            print('Equity :', _equity)

            DATE.append(price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'))
            CONTRACT.append(x)
            OPEN_POZ.append(0)
            CLOSE_POZ.append(1)
            RATE_OPEN_POZ.append(0)
            RATE_CLOSE_POZ.append(price.Close_Bid.iloc[i])
            PNL_LAT.append(0)
            PNL_REAL.append(_pnl)
            TOTAL_CLOSE.append(_open_buy) 
            TOTAL_PNL_LAT.append(0)
            TOTAL_PNL_REAL.append(_pnl)
            PRICE_BUY = []
            _open_buy = 0


    return(TRADE_DURATION,_nbtransactions,EQUITY,df_resultats,_cash_ini,_pnl,_open_buy,_open_sell)



if __name__ == "__main__":
    pass


