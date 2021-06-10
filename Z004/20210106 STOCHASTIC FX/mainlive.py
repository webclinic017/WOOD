__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
###################### LIVE ON FXCM LIVE ###########################
####################################################################
####################################################################
 

import time
import pandas as pd
import numpy as np
import colorama as col
from tqdm import tqdm
import joblib
from joblib import Parallel,delayed
import datetime as dt
import fxcmpy
import pyttsx3
import datetime as dt
from Live import *
from librairies.strategy import *
from sklearn.preprocessing import MaxAbsScaler,quantile_transform
engine = pyttsx3.init()
print('version fxcmpy :',fxcmpy.__version__)


_token = 'dbdc379ce7761772c662c3e92250a0ae38385b2c'
_server = 'demo'
_user_id = 'D261282181'
_compte = '01215060'
_password = 'waXz1'

_period = 'm15'
_name = 'MLPClassifier'

TICKER_LIST = ['USD/MXN']
x = TICKER_LIST[0]
con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)


def conX():
    con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
    if con.is_connected() == True:
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
    else:
        print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
    return(con)

def deconX(con):
    con = con.close()
    if con.is_connected() == True:
        print(col.Fore.GREEN+'Connexion non intérrompue'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
    else:
        print(col.Fore.RED+'Connexion intérrompue'+col.Style.RESET_ALL)
    return()

def scrap_hist(ticker,invers = 'non'):
    #_debut = pd.to_datetime((dt.datetime.now()-dt.timedelta(minutes=3987165)).strftime('%Y-%m-%d'))
    #_fin = pd.to_datetime((dt.datetime.now().strftime('%Y-%m-%d')))
    data = con.get_candles(ticker,period=_period,start=_debut,end=_fin)
    data['Open'] = (data['bidopen']+data['askopen'])/2
    data['High'] = (data['bidhigh']+data['askhigh'])/2
    data['Low'] = (data['bidlow']+data['asklow'])/2
    data['Close'] = (data['bidclose']+data['askclose'])/2
    return(data)

def buy(df_all):
    print(dt.datetime.now())
    ##### BUY 
    _price = round(con.get_candles(x,period='m15',number=1).askclose[-1],5)
    _time = con.get_candles(x,period='m15',number=1).index[-1]
    _amount = 50 
    _limit = round(_price + _price * 0.004,5)
    _stop = round(_price  - _price * 0.002,5)
    _atmarket = 3
    order = con.open_trade(symbol=x,is_buy=True, is_in_pips=False, amount=20, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)
    print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
    print(col.Fore.GREEN,'Achat sur le ticker',col.Fore.YELLOW,x,col.Fore.GREEN,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

def sell(df_all):
    print(dt.datetime.now())
    _atmarket = 3
    _price = round(con.get_candles(x,period='m15',number=1).bidclose[-1],5)
    _amount = 50 
    _stop = round(_price + _price * 0.002,5)
    _limit = round(_price  - _price * 0.004,5)
    order = con.open_trade(symbol=x,is_buy=False, is_in_pips=False, amount=20, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)
    print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
    print(col.Fore.RED,'Vente sur le ticker',col.Fore.YELLOW,x,col.Fore.RED,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

    #expiration = (dt.datetime.now() + dt.timedelta(hours=4)).strftime(format='%Y-%m-%d %H:%M'))
    return()

def close():
    con.close_all_for_symbol(x)
    return()

def init_df():
    _path = 'JOBLIB/Ticker_'+_period+'/df_'+x.replace('/','')
    df_all = joblib.load('JOBLIB/Built_bases/df_all')
    df_all = df_all[df_all.Symbol==x.replace('/','')]
    df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
    _fin = dt.datetime.now()
    _deb = df_all.index[-1]
    _debut = dt.datetime(_deb.year,_deb.month,_deb.day,_deb.hour,_deb.minute)
    addon = con.get_candles(x,period='m15',start=_debut,end=_fin).drop(['tickqty'],axis=1)
    addon = addon.rename(columns={'bidopen':'OpenBid','bidclose':'CloseBid','bidhigh':'HighBid','bidlow':'LowBid','askopen':'OpenAsk','askclose':'CloseAsk','askhigh':'HighAsk','asklow':'LowAsk'})
    addon['Open'] = (addon.OpenAsk + addon.OpenBid)/2
    addon['High'] = (addon.HighAsk + addon.HighBid)/2
    addon['Low'] = (addon.LowAsk + addon.LowBid)/2
    addon['Close'] = (addon.CloseAsk + addon.CloseBid)/2
    addon['Symbol'] = x.replace('/','')
    addon['Date'] = addon.index
    addon['Date'] = pd.to_datetime(addon['Date'].dt.strftime(date_format='%Y-%m-%d'))
    df_all = df_all.append(addon.iloc[1:,:])
    joblib.dump(df_all,_path)
    return(df_all)

print('Global Optimized LumberJack Environment Motor for For_Ex\nLumberJack Jyss 5781(c)')
print(col.Fore.BLUE,'°0Oo_D.A.G._26_oO0°')
print(col.Fore.YELLOW,col.Back.BLUE,'--- Bigfoot 1. #v0.60 ---',col.Style.RESET_ALL)


print('')
engine.say(" Initialization of Bigfoot 1, FX system")
engine.say("Bigfoot's Connexion to the a p i")
engine.runAndWait()

try:
    con.is_connected() == True
    
    engine.say("already Connected")
    engine.runAndWait()
    print(col.Fore.GREEN+'Connexion rétablie'+col.Style.RESET_ALL)
    print('Compte utilisé : ',con.get_account_ids())
    print('')
    
except:
    try:
        con = conX()
        con.is_connected() == True
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
        engine.say("Bigfoot is Connected")
        engine.runAndWait()
    except:
        print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
        engine.say("Mayday, mayday, Not Connected, mauzerfucker!")
        engine.say("Check your internet, and launch agin the Bigfoot")
        engine.runAndWait()
        print('')
        #os._exit(0)
        con = deconX()
        time.sleep(1)
        con = conX()
print('\rChargement de la base...',end='',flush=True)
engine.say("Ignition of Bigfoot. Loading the database.")
engine.runAndWait()
_path = 'JOBLIB/Ticker_'+_period+'/df_'+x.replace('/','')
df_all = joblib.load(_path)
df_all = df_all[df_all.Symbol==x.replace('/','')]
df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
engine.say("Database is loaded. Ready to enter Live")
engine.runAndWait()
print('\rBase Chargée.',end='',flush=True)
while True:
    while dt.datetime.now().minute not in [0,15,30,45]:
        print('\rTicker tracké :',x,' ',dt.datetime.now(),end='',flush=True)
        time.sleep(1)
    print()
    while con.get_candles(x,period=_period,start=dt.datetime(df_all.index[-1].year,df_all.index[-1].month,df_all.index[-1].day,df_all.index[-1].hour,df_all.index[-1].minute)\
          ,end=dt.datetime.now()).index[-1].minute != dt.datetime.now().minute:

          time.sleep(0.1)
        
    _fin = dt.datetime.now()
    _deb = df_all.index[-1]
    _debut = dt.datetime(_deb.year,_deb.month,_deb.day,_deb.hour,_deb.minute)
    addon = con.get_candles(x,period='m15',start=_debut,end=_fin).drop(['tickqty'],axis=1)
    addon = addon.rename(columns={'bidopen':'OpenBid','bidclose':'CloseBid','bidhigh':'HighBid','bidlow':'LowBid','askopen':'OpenAsk','askclose':'CloseAsk','askhigh':'HighAsk','asklow':'LowAsk'})
    addon['Open'] = (addon.OpenAsk + addon.OpenBid)/2
    addon['High'] = (addon.HighAsk + addon.HighBid)/2
    addon['Low'] = (addon.LowAsk + addon.LowBid)/2
    addon['Close'] = (addon.CloseAsk + addon.CloseBid)/2
    addon['Symbol'] = x.replace('/','')
    addon['Date'] = addon.index
    addon['Date'] = pd.to_datetime(addon['Date'].dt.strftime(date_format='%Y-%m-%d'))
    df_all = df_all.append(addon.iloc[1:,:])
    #df_all = df_all.iloc[-263570:,:]
    

    ##### Si la période demandée est déjà H1, on peut construire directement la base daily pour tous les tickers => daily_all
    if _period == 'H1':
        df_all = timerange1D(df_all)
        daily_all = get_daily(df_all,TICKER_LIST)

    ##### Si la période n'est pas H1, on récupère d'abord les data en 1H pour tous les tickers, et on construit la base daily à partir du 1H => daily_all
    else:
        _period='H1'
        df_all = timerange1D(df_all)
        hourly_all = df_all[df_all.index.minute==0] # scrap_hist(x)
        #hourly_all['Symbol'] = x.replace('/','')
        hourly_all = timerange1D(hourly_all)
        _period='m15'
        daily_all = get_daily(hourly_all,TICKER_LIST)
        del hourly_all
    daily_all = timerange1W(daily_all)
    weekly_all = get_weekly(daily_all,TICKER_LIST)
    daily_all = adr(daily_all,_window=14)
    df_all = getadr(daily_all,df_all,TICKER_LIST)
    df_all = adrhnl(daily_all,df_all,TICKER_LIST)
    df_all = sma(df_all=df_all,_window=200)
    df_all = bollinger(df_all,_slow=20)
    df_all = slowstochastic(df_all,TICKER_LIST)
    df_all = ema(df_all,21,TICKER_LIST)
    df_all = ema(df_all,8,TICKER_LIST)
    weekly_all = pivot(weekly_all,TICKER_LIST)
    df_all = pivotimportdf(df_all,weekly_all,TICKER_LIST)
    df_all = atr(df_all,TICKER_LIST,14)
    df_all = rvi(df_all,TICKER_LIST,_window=14)
    df_all = sbgamma(df_all,TICKER_LIST)
    df_all = onhisma(df_all,TICKER_LIST,_window=5)
    df_all = onlosma(df_all,TICKER_LIST,_window=5)
    df_all = onhisma(df_all,TICKER_LIST,_window=21)
    df_all = onlosma(df_all,TICKER_LIST,_window=21)
    df_all = onhisma(df_all,TICKER_LIST,_window=34)
    df_all = onlosma(df_all,TICKER_LIST,_window=34)
    df_all = importohlc(df_all,weekly_all,TICKER_LIST,_suffix='_weekly')
    df_all = importohlc(df_all=df_all,other_all=daily_all,TICKER_LIST=TICKER_LIST,_suffix='_daily')
    df_all = stochastic(df_all)

    features = featuring(df_all)

    # And drop the nan
    features = features.dropna()
    ##### Signal is from strategy. This is potential good one. But we have to create the TRACKER column where the Signal where efficient

    # Proceed an MaxAbsScaler on features
    features = scaling(features,scaler=MaxAbsScaler())

    features = quantile(features,quantile_transform)

    savename = 'JOBLIB/'+_name+'/Save_'+x.replace('/','')+'_m15.sav'
    _model = joblib.load(savename)

    _valid = _model.predict(features.drop(['Date','Symbol','Signal'],axis=1))[-1]

    _signal = df_all.Signal[-1]

    if _valid == 1 and _signal == 1 :
        buy()

    elif _valid == 1 and _signal == -1 :
        sell()

    else:
        print('\nnothing for',x)
    
    print()
   
    df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]  
    joblib.dump(df_all, _path)
    df_all = joblib.load(_path)



if __name__ == "__main__":
    pass 