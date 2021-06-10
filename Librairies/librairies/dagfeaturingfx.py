__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
####### RECUPERATION DONNEES ET PREPARATION DES DATA FX ############
####################################################################
####################################################################

import joblib
from joblib import Parallel,delayed
import pandas as pd
import numpy as np
import datetime as dt
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import colorama as col
import time
sys.path.append('../') 
import pyttsx3
engine = pyttsx3.init()


def get_ticker_list(): 
    print(col.Fore.BLUE+'\nRécupération des tickers'+col.Style.RESET_ALL)
    KEY_LIST = []
    RAW_LIST = os.listdir('../BASES/Base/')
    KEY_LIST = [(_cur.split('_')[0][:3]+'/'+_cur.split('_')[0][3:]) for _cur in RAW_LIST]
    for i in KEY_LIST:
        if '.' in i :
            KEY_LIST.remove(i)
    return(list(set(sorted(KEY_LIST))))

def get_all_data(TICKER_LIST,_period):
    print(col.Fore.YELLOW+'\nRécupération des data intraday'+col.Style.RESET_ALL)
    def funk(_ticker,_period):
        _ticker = _ticker.replace('/','')
        
        globals()['df_%s' %_ticker] = pd.DataFrame()

        print(col.Fore.BLUE,'Ticker',col.Fore.YELLOW,_ticker[:3]+'/'+_ticker[3:],col.Style.RESET_ALL)
        ##### Chargement de la base par ticker
        globals()['df_%s' %_ticker] = pd.read_csv('../BASES/Base/'+_ticker+'_'+_period+'_BidAndAsk.csv')

        ##### Ajout de la colonne Symbol pour identifier le ticker
        globals()['df_%s' %_ticker]['Symbol'] = _ticker

        ##### On fixe la date en index sous forme de Timestamp
        globals()['df_%s' %_ticker]['Lindex'] = pd.to_datetime(globals()['df_%s' %_ticker]['Date'] + ' ' + globals()['df_%s' %_ticker]['Time'])
        globals()['df_%s' %_ticker].set_index(pd.to_datetime(globals()['df_%s' %_ticker].Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)

        ###### On drop les colonnes inutiles
        globals()['df_%s' %_ticker] = globals()['df_%s' %_ticker].drop(['Date','Lindex','Time','Total Ticks'],axis=1)

        ##### On enlève les jours correspondant au samedi et au dimanche
        globals()['df_%s' %_ticker]['WE'] = np.where(((globals()['df_%s' %_ticker].index.weekday == 5) | (globals()['df_%s' %_ticker].index.weekday == 6)),None,globals()['df_%s' %_ticker].index.weekday)
        globals()['df_%s' %_ticker] = globals()['df_%s' %_ticker].dropna()
        globals()['df_%s' %_ticker] = globals()['df_%s' %_ticker].drop(['WE'],axis=1)

        ##### Calcul des averages pour les OHLC
        globals()['df_%s' %_ticker]['Open'] = (globals()['df_%s' %_ticker]['OpenBid'] + globals()['df_%s' %_ticker]['OpenAsk']) / 2
        globals()['df_%s' %_ticker]['High'] = (globals()['df_%s' %_ticker]['HighBid'] + globals()['df_%s' %_ticker]['HighAsk']) / 2
        globals()['df_%s' %_ticker]['Low'] = (globals()['df_%s' %_ticker]['LowBid'] + globals()['df_%s' %_ticker]['LowAsk']) / 2
        globals()['df_%s' %_ticker]['Close'] = (globals()['df_%s' %_ticker]['CloseBid'] + globals()['df_%s' %_ticker]['CloseAsk']) / 2

        return(globals()['df_%s' %_ticker].sort_index(axis=1))
    df_all = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(funk)(_ticker=_ticker,_period=_period) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    df_all = df_all.append(parallel_pool(delayed_funcs))
    return(df_all.sort_index(axis=1))

def get_daily(df_all,TICKER_LIST):
    print(col.Fore.GREEN+'\nRécupération des data daily'+col.Style.RESET_ALL)
    def funky(df_all,_ticker):
        _ticker = _ticker.replace('/','')
        globals()['df_%s' %_ticker] = df_all[df_all.Symbol == _ticker]
        globals()['daily_%s' %_ticker] = pd.DataFrame(index=globals()['df_%s' %_ticker].Date.unique())

        print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,_ticker[:3]+'/'+_ticker[3:],col.Style.RESET_ALL,end='',flush=True)

        ##### Fabrication de la base daily
        globals()['daily_%s' %_ticker]['Lindex'] = list((globals()['df_%s' %_ticker].groupby('Date').Date.first()))
        globals()['daily_%s' %_ticker]['Open'] = list((globals()['df_%s' %_ticker].groupby('Date').Open.first()))
        globals()['daily_%s' %_ticker]['High'] = list((globals()['df_%s' %_ticker].groupby('Date').High.max()))
        globals()['daily_%s' %_ticker]['Low'] = list((globals()['df_%s' %_ticker].groupby('Date').Low.min()))
        globals()['daily_%s' %_ticker]['Close'] = list((globals()['df_%s' %_ticker].groupby('Date').Close.last()))
        globals()['daily_%s' %_ticker]['Symbol'] = _ticker
        globals()['daily_%s' %_ticker] = globals()['daily_%s' %_ticker].sort_values('Lindex') ##########
        globals()['daily_%s' %_ticker].set_index(pd.to_datetime(globals()['daily_%s' %_ticker].Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True) #####
        globals()['daily_%s' %_ticker]['Date'] = globals()['daily_%s' %_ticker].Lindex
        globals()['daily_%s' %_ticker] = globals()['daily_%s' %_ticker].drop(['Lindex'],axis=1)
        #globals()['daily_%s' %_ticker] = globals()['daily_%s' %_ticker].sort_values('Lindex')
        #globals()['daily_%s' %_ticker].set_index(pd.to_datetime(globals()['daily_%s' %_ticker].Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)

        return(globals()['daily_%s' %_ticker].sort_index(axis=1))
    daily_all = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(funky)(df_all=df_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    daily_all = daily_all.append(parallel_pool(delayed_funcs))
    #daily_all = daily_all.drop(['Lindex'],axis=1)
    return(daily_all.sort_index(axis=0))

def get_weekly(daily_all,TICKER_LIST):
    print(col.Fore.BLUE+'\nRécupération des tickers weekly'+col.Style.RESET_ALL)
    def funkette(daily_all,_ticker):
        _ticker = _ticker.replace('/','')
        globals()['df_%s' %_ticker] = daily_all[daily_all.Symbol == _ticker]
        globals()['weekly_%s' %_ticker] = pd.DataFrame()

        print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,_ticker[:3]+'/'+_ticker[3:],col.Style.RESET_ALL,end='',flush=True)

        globals()['weekly_%s' %_ticker]['Lindex'] = list((globals()['df_%s' %_ticker].groupby('Week').Date.first()))
        globals()['weekly_%s' %_ticker]['Open'] = list((globals()['df_%s' %_ticker].groupby('Week').Open.first()))
        globals()['weekly_%s' %_ticker]['High'] = list((globals()['df_%s' %_ticker].groupby('Week').High.max()))
        globals()['weekly_%s' %_ticker]['Low'] = list((globals()['df_%s' %_ticker].groupby('Week').Low.min()))
        globals()['weekly_%s' %_ticker]['Close'] = list((globals()['df_%s' %_ticker].groupby('Week').Close.last()))
        globals()['weekly_%s' %_ticker] = globals()['weekly_%s' %_ticker].sort_values('Lindex')
        globals()['weekly_%s' %_ticker].set_index(pd.to_datetime(globals()['weekly_%s' %_ticker].Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)
        globals()['weekly_%s' %_ticker]['Symbol'] = _ticker
        globals()['weekly_%s' %_ticker]['Date'] = globals()['weekly_%s' %_ticker].Lindex
        globals()['weekly_%s' %_ticker] = globals()['weekly_%s' %_ticker].drop(['Lindex'],axis=1)
        
        return(globals()['weekly_%s' %_ticker].sort_index(axis=0))

    daily_all['WeekDay'] = np.where(daily_all.sort_values('Symbol').Week!=daily_all.sort_values('Symbol').Week.shift(1),daily_all.index,np.datetime64('NaT'))
    weekly_all = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(funkette)(daily_all=daily_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    weekly_all = weekly_all.append(parallel_pool(delayed_funcs))
    return(weekly_all.sort_index(axis=0))

def timerange1D(df_all):
    print('\nAjout Date')
    df_all['Date'] = df_all.index
    df_all['Date'] = df_all['Date'].dt.strftime(date_format='%Y-%m-%d')
    return(df_all.sort_index(axis=0))

def timerange1W(daily_all):
    print("\nAjout colonne 'Date dans le weekly" ) 
    daily_all['WeekNo'] = pd.to_datetime(daily_all.index)
    daily_all['WeekNo'] = daily_all['WeekNo'].dt.week.astype(str)
    daily_all['Year'] = pd.to_datetime(daily_all.index)
    daily_all['Year'] = daily_all['Year'].dt.year.astype(str)
    daily_all['Date'] = pd.to_datetime(daily_all.index)
    daily_all['list']=daily_all[['Year','WeekNo']].values.tolist()
    daily_all['Week']=daily_all['list'].apply('_'.join)
    #daily_all.sort_values('Date')
    return(daily_all.sort_index(axis=0))

def Wilder(data, window):
    
    '''Smoothening or moving average that is commonly used with other indicators. 
    Although SMA is quite common, it contains a bias of giving equal weight to each value in the past. 
    To solve this, Wells Wilder introduced a new version of smoothening that places more weight on the recent events. 
    We will use Wilder’s Smoothing for most of our following indicators, and below is the function that can be generally used to obtain this Smoothing.'''

    start = np.where(~np.isnan(data))[0][0] # Positionne après les nan
    Wilder = np.array([np.nan]*len(data)) # Replace les nan en début de liste pour ne pas changer la longueur
    Wilder[start+window-1] = data[start:(start+window)].mean() #Simple Moving Average pour la window window
    for i in range(start+window,len(data)):
        Wilder[i] = ((Wilder[i-1]*(window-1) + data[i])/window) #Wilder Smoothing
    return(Wilder)

def ema(df_all, _window,TICKER_LIST):
    print(col.Fore.GREEN+"\nCalcul de l'EMA_"+str(_window)+"en cours"+col.Style.RESET_ALL)
    
    def slice_ema(df_all,_window,_ticker):
        _ticker = _ticker.replace('/','')
        df = df_all[df_all.Symbol==_ticker]
        df['EMA_'+str(_window)] = df.Close.ewm(span=_window,adjust=False).mean()
        return(df)
    
    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(slice_ema)(df_all=df_all,_window=_window,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    df_temp = df_temp.append(parallel_pool(delayed_funcs))
    engine.say("Bim, the job is done")
    engine.runAndWait()
    print(col.Fore.GREEN+"\nProcessus de calcul de l'EMA_"+str(_window)+"terminé"+col.Style.RESET_ALL)
    return(df_temp.sort_index(axis=0))

def smaratio(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul SMA'+col.Style.RESET_ALL)
    '''Simple Moving Average (SMA)
    Simple Moving Average is one of the most common technical indicators. 
    SMA calculates the average of prices over a given interval of time and is used to determine the trend of the stock. 
    As defined above, I will create a slow SMA (SMA_15) and a fast SMA (SMA_5). 
    To provide Machine Learning algorithms with already engineered factors, 
    one can also use (SMA_15/SMA_5) or (SMA_15 - SMA_5) as a factor to capture the relationship between these two moving averages.
    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''


    df_all['SMA_'+str(_fast)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = _fast).mean())
    df_all['SMA_'+str(_slow)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = _slow).mean())
    df_all['SMA_ratio'] = df_all['SMA_'+str(_slow)] / df_all['SMA_'+str(_fast)]
    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))].Close)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['SMA_'+str(_slow)])
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['SMA_'+str(_fast)])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close','SMA_'+str(_slow),'SMA_'+str(_fast)],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Price, Slow and Fast Moving Average from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['SMA_ratio'], color = 'blue')
        ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()
    return(df_all.sort_index(axis=0))

def sma(df_all,_window=200,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul SMA'+col.Style.RESET_ALL)
    '''Simple Moving Average (SMA)
    Simple Moving Average is one of the most common technical indicators. 
    SMA calculates the average of prices over a given interval of time and is used to determine the trend of the stock. 
    As defined above, I will create a slow SMA (SMA_15) and a fast SMA (SMA_5). 
    To provide Machine Learning algorithms with already engineered factors, 
    one can also use (SMA_15/SMA_5) or (SMA_15 - SMA_5) as a factor to capture the relationship between these two moving averages.
    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''


    df_all['SMA_'+str(_window)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = _window).mean())
    
    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))].Close)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['SMA_'+str(_window)])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close','SMA_'+str(_window)],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Price, Simple Moving Average from "+str(start)+' to '+str(end), fontsize = 20)
        plt.show()
    return(df_all.sort_index(axis=0))

def atrratio(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul ATR RATIO'+col.Style.RESET_ALL)
    '''Average True Range is a common technical indicator used to measure volatility in the market, measured as a moving average of True Ranges. 
    A higher ATR of a company implied higher volatility of the stock. 
    ATR however is primarily used in identifying when to exit or enter a trade rather than the direction in which to trade the stock.
    As defined above, a slow ATR represents 5 days moving average and fast ATR represents 15 days moving average.
    True Range is defined as maximum of:
    a. High - Low
    b. abs(High - Previous Close)
    c. abs(Low - Previous Close)
    
    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### On relève le close de la bougie précédente
    df_all['prev_close'] = df_all.groupby('Symbol')['Close'].shift(1)

    ##### On récupère le maximum parmi 3 possibilités :
        ##### High - Low
        ##### High moins close précédent
        ##### Close précédent - Low
    df_all['TR'] = np.maximum((df_all['High'] - df_all['Low']), 
                        np.maximum(abs(df_all['High'] - df_all['prev_close']), 
                        abs(df_all['prev_close'] - df_all['Low'])))
    
    ##### (TODO : Multiprocessing sur le for loop)
    for i in df_all['Symbol'].unique():
        print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,i,col.Style.RESET_ALL,end='',flush=True)
        TR_data = df_all[df_all.Symbol == i].copy()
        df_all.loc[df_all.Symbol==i,'ATR_'+str(_fast)] = Wilder(TR_data['TR'], _fast)
        df_all.loc[df_all.Symbol==i,'ATR_'+str(_slow)] = Wilder(TR_data['TR'], _slow)

    df_all['ATR_Ratio'] = df_all['ATR_'+str(_fast)] / df_all['ATR_'+str(_slow)]
    
    df_all = df_all.drop(['prev_close','TR'],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['ATR_'+str(_slow)])
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['ATR_'+str(_fast)])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['ATR_'+str(_slow),'ATR_'+str(_fast)],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Average True Range from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['ATR_Ratio'], color = 'blue')
        ax1.legend(['ATR_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()
    return(df_all.sort_index(axis=0))

def adx(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul ADX'+col.Style.RESET_ALL)
    '''Average Directional Index (ADX)
    Average Directional Index was developed by Wilder to assess the strength of a trend in stock prices. 
    Two of its main components, +DI and -DI helps in identifying the direction of the trend. 
    In general, an ADX of 25 or above indicates a strong trend and an ADX of less than 20 indicates a weak trend. 
    The calculation of ADX is quite complex and requires certain steps.
    
    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### On récupère les High et Low de la bougie d'avant
    df_all['prev_high'] = df_all.groupby('Symbol')['High'].shift(1)
    df_all['prev_low'] = df_all.groupby('Symbol')['Low'].shift(1)

    ##### tilde[option + n] = complement operator ou inverse.

    ##### Pour +DM
    ##### Si le prev Hign n'est pas nan : 
        ##### Si le High > prev High ET QUE (High-prev High) > (Prev Low - Low) => on met (High - Prev High), sinon 0.
    ##### Si le prev High etait un nan, on met nan
    df_all['+DM'] = np.where(~np.isnan(df_all.prev_high),
                            np.where((df_all['High'] > df_all['prev_high']) & 
            (((df_all['High'] - df_all['prev_high']) > (df_all['prev_low'] - df_all['Low']))), 
                                                                    df_all['High'] - df_all['prev_high'], 
                                                                   0),np.nan)
    ##### Pour -DM 
    ##### Si le prev Low n'est pas nan : 
        ##### Si le Prev Low > Low ET QUE (Prev Low - Low) > (High - Prev High) => on met (Prev Low - Low), sinon 0.
    ##### Si le prev High etait un nan, on met nan
    df_all['-DM'] = np.where(~np.isnan(df_all.prev_low),
                            np.where((df_all['prev_low'] > df_all['Low']) & 
            (((df_all['prev_low'] - df_all['Low']) > (df_all['High'] - df_all['prev_high']))), 
                                        df_all['prev_low'] - df_all['Low'], 
                                        0),np.nan)

    ##### On passe pour chaque Symbol (TODO : Multiprocessing sur le for loop)
    ##### On créé had hoc un Array ADX_data qui est une copy() de df_all
    ##### Pour +DM et -DM, on récupère le Wilder fast et slow, et on remplace les valeurs de +DM et -DM
    for i in df_all['Symbol'].unique():
        print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,i,col.Style.RESET_ALL,end='',flush=True)
        ADX_data = df_all[df_all.Symbol == i].copy()
        df_all.loc[df_all.Symbol==i,'+DM_'+str(_fast)] = Wilder(ADX_data['+DM'], _fast)
        df_all.loc[df_all.Symbol==i,'-DM_'+str(_fast)] = Wilder(ADX_data['-DM'], _fast)
        df_all.loc[df_all.Symbol==i,'+DM_'+str(_slow)] = Wilder(ADX_data['+DM'], _slow)
        df_all.loc[df_all.Symbol==i,'-DM_'+str(_slow)] = Wilder(ADX_data['-DM'], _slow)

    ##### On créé alors +DI et -DI fast et slow en divisant +DM et -DM par l'ATR fast ou slow, selon le cas idoine.
    df_all['+DI_'+str(_fast)] = (df_all['+DM_'+str(_fast)]/df_all['ATR_'+str(_fast)])*100
    df_all['-DI_'+str(_fast)] = (df_all['-DM_'+str(_fast)]/df_all['ATR_'+str(_fast)])*100
    df_all['+DI_'+str(_slow)] = (df_all['+DM_'+str(_slow)]/df_all['ATR_'+str(_slow)])*100
    df_all['-DI_'+str(_slow)] = (df_all['-DM_'+str(_slow)]/df_all['ATR_'+str(_slow)])*100

    ##### On peut alors calculer les DX fast et slow en calculant dans chaque cas (+DI - -DI)/(+DI + -DI)
    df_all['DX_'+str(_fast)] = (np.round(abs(df_all['+DI_'+str(_fast)] - df_all['-DI_'+str(_fast)])/(df_all['+DI_'+str(_fast)] + df_all['-DI_'+str(_fast)]) * 100))

    df_all['DX_'+str(_slow)] = (np.round(abs(df_all['+DI_'+str(_slow)] - df_all['-DI_'+str(_slow)])/(df_all['+DI_'+str(_slow)] + df_all['-DI_'+str(_slow)]) * 100))

    ##### On passe pour chaque Symbol (TODO : Multiprocessing sur le for loop)
    ##### On créé had hoc un Array ADX_data qui est une copy() de df_all
    ##### On créé les ADX slow et fast en passant les DX au smoothering du Wilder
    for i in df_all['Symbol'].unique():
        print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,i,col.Style.RESET_ALL,end='',flush=True)
        ADX_data = df_all[df_all.Symbol == i].copy()
        df_all.loc[df_all.Symbol==i,'ADX_'+str(_fast)] = Wilder(ADX_data['DX_'+str(_fast)], _fast)
        df_all.loc[df_all.Symbol==i,'ADX_'+str(_slow)] = Wilder(ADX_data['DX_'+str(_slow)], _slow)

    df_all = df_all.drop(['DX_'+str(_fast),'DX_'+str(_slow),'+DI_'+str(_fast),'-DI_'+str(_fast),'+DI_'+str(_slow),'-DI_'+str(_slow),'-DM','+DM','prev_high','prev_low'],axis=1)


    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['ADX_'+str(_fast)], color = 'blue')
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['ADX_'+str(_slow)], color = 'orange')
        ax1.legend(['ADX_'+str(_fast),'ADX_'+str(_slow)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()

    return(df_all.sort_index(axis=0))

def slowstochastic(df_all,TICKER_LIST,_window=5,_per=3,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.GREEN+'\nCalcul du Slow STOCHASTIC'+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Fenêtre de Glissement : "+col.Fore.YELLOW+str(_window)+" periodes."+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Période de lissage : "+col.Fore.YELLOW+str(_per)+" periodes."+col.Style.RESET_ALL)
    def slowstok(df_all,_window,_per,_ticker):
        _ticker = _ticker.replace('/','')
        df_1 = df_all[df_all.Symbol==_ticker]
        '''Stochastic Oscillators slow version
        Stochastic oscillator is a momentum indicator aiming at identifying overbought 
            AND oversold securities 
            AND is commonly used in technical analysis.
            
        df_all = La base à travailler, _window = fenetre , _per = periode pour le smoothering
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

        ##### On va récupérer pour les fenetres fast et slow les plus bas Low et les plus hauts High
        df_1['Lowest_'+str(_window)] = df_1['Low'].transform(lambda x: x.rolling(window = _window).min())
        df_1['Highest_'+str(_window)] = df_1['High'].transform(lambda x: x.rolling(window = _window).max())


        ##### On calcule alors en slow et fast le stochastic
        ##### (Close - Lowest) / (Highest - Lowest)
        df_1['slow_K'+str(_window)] = (((df_1['Close'] - df_1['Lowest_'+str(_window)])/(df_1['Highest_'+str(_window)] - df_1['Lowest_'+str(_window)]))*100).rolling(window = _per).mean()

        ##### On smoothering le stochastic en calculant la moyenne sur les fenetres slow et fast de ces valeurs
        df_1['slow_D'+str(_window)] = df_1['slow_K'+str(_window)].rolling(window = _per).mean()

        df_1 = df_1.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)

        return(df_1.sort_index(axis=0))
    
    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(slowstok)(df_all=df_all,_window=_window,_per=_per,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    df_temp = df_temp.append(parallel_pool(delayed_funcs))

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['slow_K'+str(_window)], color = 'blue')
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['slow_D'+str(_window)], color = 'orange')
        ax1.legend(['slow_K'+str(_window),'slow_D'+str(_window)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adj
    
    return(df_temp.sort_index(axis=0))

def faststochastic(df_all,TICKER_LIST,_window=5,_per=3,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.GREEN+'\nCalcul du Slow STOCHASTIC'+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Fenêtre de Glissement : "+col.Fore.YELLOW+str(_window)+" periodes."+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Période de lissage : "+col.Fore.YELLOW+str(_per)+" periodes."+col.Style.RESET_ALL)
    def faststok(df_all,_window,_per,_ticker):
        _ticker = _ticker.replace('/','')
        df_1 = df_all[df_all.Symbol==_ticker]
        '''Stochastic Oscillators slow version
        Stochastic oscillator is a momentum indicator aiming at identifying overbought 
            AND oversold securities 
            AND is commonly used in technical analysis.
            
        df_all = La base à travailler, _window = fenetre , _per = periode pour le smoothering
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

        ##### On va récupérer pour les fenetres fast et slow les plus bas Low et les plus hauts High
        df_1['Lowest_'+str(_window)] = df_1['Low'].transform(lambda x: x.rolling(window = _window).min())
        df_1['Highest_'+str(_window)] = df_1['High'].transform(lambda x: x.rolling(window = _window).max())

        ##### On calcule alors en slow et fast le stochastic
        ##### (Close - Lowest) / (Highest - Lowest)
        df_1['fast_K'+str(_window)] = (((df_1['Close'] - df_1['Lowest_'+str(_window)])/(df_1['Highest_'+str(_window)] - df_1['Lowest_'+str(_window)]))*100)

        ##### On smoothering le stochastic en calculant la moyenne sur les fenetres slow et fast de ces valeurs
        df_1['fast_D'+str(_window)] = df_1['slow_K'+str(_window)].rolling(window = _per).mean()
        df_1 = df_1.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)
        return(df_1.sort_index(axis=0))
    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(faststok)(df_all=df_all,_window=_window,_per=_per,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    df_temp = df_temp.append(parallel_pool(delayed_funcs))

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['fast_K'+str(_window)], color = 'blue')
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['fast_D'+str(_window)], color = 'orange')
        ax1.legend(['fast_K'+str(_window),'fast_D'+str(_window)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()
    

    return(df_temp.sort_index(axis=0))

def fullstochastic(df_all,TICKER_LIST,_window=5,_per1=3,_per2=3,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.GREEN+'\nCalcul du Slow STOCHASTIC'+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Fenêtre de Glissement : "+col.Fore.YELLOW+str(_window)+" periodes."+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Période de lissage : "+col.Fore.YELLOW+str(_per)+" periodes."+col.Style.RESET_ALL)
    def fullstok(df_all,_window,_per1,_per2,_ticker):
        _ticker = _ticker.replace('/','')
        df_1 = df_all[df_all.Symbol==_ticker]
        '''Stochastic Oscillators slow version
        Stochastic oscillator is a momentum indicator aiming at identifying overbought 
            AND oversold securities 
            AND is commonly used in technical analysis.
            
        df_all = La base à travailler, _window = fenetre , _per = periode pour le smoothering
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

        ##### On va récupérer pour les fenetres fast et slow les plus bas Low et les plus hauts High
        df_1['Lowest_'+str(_window)] = df_1['Low'].transform(lambda x: x.rolling(window = _window).min())
        df_1['Highest_'+str(_window)] = df_1['High'].transform(lambda x: x.rolling(window = _window).max())


        ##### On calcule alors en slow et fast le stochastic
        ##### (Close - Lowest) / (Highest - Lowest)
        df_1['full_K'+str(_window)] = (((df_1['Close'] - df_1['Lowest_'+str(_window)])/(df_1['Highest_'+str(_window)] - df_1['Lowest_'+str(_window)]))*100).rolling(window = _per1).mean()

        ##### On smoothering le stochastic en calculant la moyenne sur les fenetres slow et fast de ces valeurs
        df_1['full_D'+str(_window)] = df_1['slow_K'+str(_window)].rolling(window = _per2).mean()

        df_1 = df_1.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)

        return(df_1.sort_index(axis=0))
    
    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(fullstok)(df_all=df_all,_window=_window,_per1=_per1,_per2=_per2,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    df_temp = df_temp.append(parallel_pool(delayed_funcs))

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['full_K'+str(_window)], color = 'blue')
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['full_D'+str(_window)], color = 'orange')
        ax1.legend(['full_K'+str(_window),'full_D'+str(_window)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adj

    return(df_temp.sort_index(axis=0))


    print(col.Fore.GREEN+'\nCalcul des slow, fast, full STOCHASTIC'+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Fenêtre de Glissement : "+col.Fore.YELLOW+str(_window)+" periodes."+col.Style.RESET_ALL)
    print(col.Fore.BLUE+"Période de lissage : "+col.Fore.YELLOW+str(_per)+" periodes."+col.Style.RESET_ALL)
    '''Stochastic Oscillators slow version
    Stochastic oscillator is a momentum indicator aiming at identifying overbought 
        AND oversold securities 
        AND is commonly used in technical analysis.
        
    df_all = La base à travailler, _window = fenetre , _per = periode pour le smoothering
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### On va récupérer pour les fenetres fast et slow les plus bas Low et les plus hauts High
    df_all['Lowest_'+str(_window)] = df_all.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window = _window).min())
    df_all['Highest_'+str(_window)] = df_all.groupby('Symbol')['High'].transform(lambda x: x.rolling(window = _window).max())
    

    ##### On calcule alors en slow et fast le stochastic
    ##### (Close - Lowest) / (Highest - Lowest)
    df_all['full_K'+str(_window)] = (((df_all['Close'] - df_all['Lowest_'+str(_window)])/(df_all['Highest_'+str(_window)] - df_all['Lowest_'+str(_window)]))*100).rolling(window = _per1).mean()

    ##### On smoothering le stochastic en calculant la moyenne sur les fenetres slow et fast de ces valeurs
    df_all['full_D'+str(_window)] = df_all['full_K'+str(_window)].rolling(window = _per2).mean()


    df_all = df_all.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Stochastic_'+str(_fast)], color = 'blue')
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Stochastic_'+str(_slow)], color = 'orange')
        ax1.legend(['STOCH_'+str(_fast),'STOCH_'+str(_slow)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()
    

    return(df_all.sort_index(axis=0))

def rsiratio(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul RSI'+col.Style.RESET_ALL)
    '''Relative Strength Index (RSI)
    RSI is one of the most common momentum indicator aimed at quantifies price changes and the speed of such change.

    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### Pour chaque Symbol, Calcule la différence du close de la cellule précédente à la cellule actuelle
    df_all['Diff'] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.diff())
    ##### Ne garde que les valeurs positives et met 0 sinon
    df_all['Up'] = df_all['Diff']
    df_all.loc[(df_all['Up']<0), 'Up'] = 0
    ##### Pour chaque Symbol, Calcule la différence du close de la cellule précédente à la cellule actuelle
    df_all['Down'] = df_all['Diff']
    ##### Ne garde que les valeurs négatives et met 0 sinon. Passe ensuite les valeurs négatives en valeur absolue
    df_all.loc[(df_all['Down']>0), 'Down'] = 0 
    df_all['Down'] = abs(df_all['Down'])

    ##### Calcule sur les fast & slow les moyennes des UP est DOWN créés
    df_all['avg_up'+str(_fast)] = df_all.groupby('Symbol')['Up'].transform(lambda x: x.rolling(window=_fast).mean())
    df_all['avg_down'+str(_fast)] = df_all.groupby('Symbol')['Down'].transform(lambda x: x.rolling(window=_fast).mean())

    df_all['avg_up'+str(_slow)] = df_all.groupby('Symbol')['Up'].transform(lambda x: x.rolling(window=_slow).mean())
    df_all['avg_down'+str(_slow)] = df_all.groupby('Symbol')['Down'].transform(lambda x: x.rolling(window=_slow).mean())

    ##### Pour les fast & slow, calcule le ratio de (moyenne UP / moyenne DOWN)
    df_all['RS_'+str(_fast)] = df_all['avg_up'+str(_fast)] / df_all['avg_down'+str(_fast)]
    df_all['RS_'+str(_slow)] = df_all['avg_up'+str(_slow)] / df_all['avg_down'+str(_slow)]

    ##### Le RSI fast & slow peut alors être calculé
    ##### 100 - (100/(1 + RS))
    df_all['RSI_'+str(_fast)] = 100 - (100/(1+df_all['RS_'+str(_fast)]))
    df_all['RSI_'+str(_slow)] = 100 - (100/(1+df_all['RS_'+str(_slow)]))

    df_all['RSI_ratio'] = df_all['RSI_'+str(_fast)]/df_all['RSI_'+str(_slow)]

    df_all = df_all.drop(['Diff','Up','Down','avg_up'+str(_fast),'avg_down'+str(_fast),'avg_up'+str(_slow),'avg_down'+str(_slow),'RS_'+str(_fast),'RS_'+str(_slow)],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['RSI_'+str(_fast)], color = 'blue')
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['RSI_'+str(_slow)], color = 'orange')
        ax1.legend(['RSI_'+str(_fast),'RSI_'+str(_slow)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()

    return(df_all.sort_index(axis=0))

def rsi(df_all,_window=5,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul RSI'+col.Style.RESET_ALL)
    '''Relative Strength Index (RSI)
    RSI is one of the most common momentum indicator aimed at quantifies price changes and the speed of such change.

    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### Pour chaque Symbol, Calcule la différence du close de la cellule précédente à la cellule actuelle
    df_all['Diff'] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.diff())
    ##### Ne garde que les valeurs positives et met 0 sinon
    df_all['Up'] = df_all['Diff']
    df_all.loc[(df_all['Up']<0), 'Up'] = 0
    ##### Pour chaque Symbol, Calcule la différence du close de la cellule précédente à la cellule actuelle
    df_all['Down'] = df_all['Diff']
    ##### Ne garde que les valeurs négatives et met 0 sinon. Passe ensuite les valeurs négatives en valeur absolue
    df_all.loc[(df_all['Down']>0), 'Down'] = 0 
    df_all['Down'] = abs(df_all['Down'])

    ##### Calcule sur les fast & slow les moyennes des UP est DOWN créés
    df_all['avg_up'+str(_window)] = df_all.groupby('Symbol')['Up'].transform(lambda x: x.rolling(window=_window).mean())
    df_all['avg_down'+str(_window)] = df_all.groupby('Symbol')['Down'].transform(lambda x: x.rolling(window=_window).mean())

    ##### Pour les fast & slow, calcule le ratio de (moyenne UP / moyenne DOWN)
    df_all['RS_'+str(_window)] = df_all['avg_up'+str(_window)] / df_all['avg_down'+str(_window)]

    ##### Le RSI fast & slow peut alors être calculé
    ##### 100 - (100/(1 + RS))
    df_all['RSI_'+str(_window)] = 100 - (100/(1+df_all['RS_'+str(_window)]))

    df_all = df_all.drop(['Diff','Up','Down','avg_up'+str(_window),'avg_down'+str(_window),'RS_'+str(_window)],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['RSI_'+str(_window)], color = 'blue')
        ax1.legend(['RSI_'+str(_window)],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()

    return(df_all.sort_index(axis=0))

def macd(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul MACD'+col.Style.RESET_ALL)
    '''Moving Average Convergence Divergence (MACD)
    MACD uses two exponentially moving averages and creates a trend analysis based on their convergence or divergence. 
    Although most commonly used MACD slow and fast signals are based on 26 days and 12 days respectively, 
    I have used 15 days and 5 days to be consistent with other indicators.

    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### Pour chaque fast & slow, et pour chaque Symbol, on calcule la moyenne mobile du close
    df_all['Ewm'+str(_fast)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=_fast, adjust=False).mean())
    df_all['Ewm'+str(_slow)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=_slow, adjust=False).mean())
    df_all['MACD'] = df_all['Ewm'+str(_slow)] - df_all['Ewm'+str(_fast)]

    df_all = df_all.drop(['Ewm'+str(_fast),'Ewm'+str(_slow)],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['MACD'], color = 'blue')
        ax1.legend(['MACD'],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()

    return(df_all.sort_index(axis=0))

def bollinger(df_all,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul BOLLINGER'+col.Style.RESET_ALL)
    '''Bollinger Bands
    Bollinger bands capture the volatility of a stock and are used to identify overbought and oversold stocks. 
    Bollinger bands consists of three main elements: The simple moving average line, 
    an upper bound which is 2 standard deviations above moving average and a lower bound which is 2 standard deviations below moving average.

    df_all = La base à travailler, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    df_all['MA'+str(_slow)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=_slow).mean())
    df_all['SD'] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=_slow).std())
    df_all['UpperBand'] = df_all['MA'+str(_slow)] + 2*df_all['SD']
    df_all['LowerBand'] = df_all['MA'+str(_slow)] - 2*df_all['SD']

    df_all = df_all.drop(['MA'+str(_slow),'SD'],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['SMA_'+str(_slow)])
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['UpperBand'])
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))].LowerBand)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))].Close)
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['SMA_'+str(_slow),'UpperBand','LowerBand','Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Price, SMA and Bollinger Bands from "+str(start)+' to '+str(end), fontsize = 20)
        plt.show()

    return(df_all.sort_index(axis=0))

def rc(df_all,_slow=15,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul RATE OF CHANGE'+col.Style.RESET_ALL)
    '''Rate of Change
    Rate of change is a momentum indicator that explains a price momentum relative to a price fixed period before.
    
    df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plo'''

    df_all['RC'] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods = _slow)) 

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['Close'])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close'],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Close from "+str(start)+' to '+str(end), fontsize = 20)

        ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
        ax1.plot(df_all[(df_all.index<=end)&(df_all.index>=start)&(df_all.Symbol==_ticker.replace('/',''))]['RC'], color = 'blue')
        ax1.legend(['Rate Of Change'],ncol=3, loc = 'upper left', fontsize = 12)
        ax1.set_facecolor('silver')
        plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        plt.show()

    return(df_all.sort_index(axis=0))

def jyss_oscillator(df_all):
    print(col.Fore.BLUE+'\nCalcul du '+col.Fore.YELLOW+'JYSS'+col.Fore.GREEN+' OSCILLATOR'+col.Style.RESET_ALL)
    print('\nCréation du Jyss Oscillator')
    print("HiWin => plus haut High d'une fenêtre de 10")
    print("LoWin => plus bas Low d'une fenêtre de 10")
    print("JyssOscBear => (HiWin - HiWin.shift(2))/HiWin.shift(2")
    print("JyssOscBull => (LoWin - LoWin.shift(2))/LoWin.shift(2")
    print('JyssOscBearSD et JyssOscBullSD => Les std() sur des fenêtres de 20')
    print('TriggerBear => (HiWin - HiWin.shift(2) * 1000')
    print('TriggerBull => (LoWin - LoWin.shift(2) * 1000')
    print('Trigger => 1 quand il y a égalité entre LoWin et LoWin.shift(9) ou quand il y a égalité entre HiWin et HiWin.shift(9')
    print('SigJyssOsc => 1 quand (Close <= LoWin) & (JyssOscUP<= -4 * JyssOscUpSD ) ou -1 quand (Close >= HiWin) & (JyssOscDwn >= 4 * JyssOscDwnSD )')
    print('')

    def jyss_osc(_ticker,df_all):
        df = pd.DataFrame()
        df['HiWin'] = df_all[df_all.Symbol==_ticker].High.rolling(10).max()
        df['LoWin'] = df_all[df_all.Symbol==_ticker].Low.rolling(10).min()
        df['JyssOscBear'] = ((df.HiWin-df.HiWin.shift(2))/df.HiWin.shift(2))*1000
        df['JyssOscBearSD'] = df.JyssOscBear.rolling(20).std()
        df['JyssOscBull'] = ((df.LoWin-df.LoWin.shift(2))/df.LoWin.shift(2))*1000
        df['JyssOscBullSD'] = df.JyssOscBull.rolling(20).std()
        df['TriggerBear'] = (df.HiWin - df.HiWin.shift(2)) * 1000
        df['TriggerBull'] = (df.LoWin - df.LoWin.shift(2)) * 1000
        df['Trigger'] = np.where((df.LoWin - df.LoWin.shift(9) == 0),1,np.where((df.HiWin - df.HiWin.shift(9) == 0 ),1,0))
        return(df)

    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(jyss_osc)(_ticker=_ticker,df_all=df_all) for _ticker in tqdm(df_all.Symbol.unique())]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    df_temp = df_temp.append(parallel_pool(delayed_funcs))
    df_all = pd.concat((df_all,df_temp),axis=1)
    return(df_all.sort_index(axis=0))

def rvi(df_all,TICKER_LIST,_window):
    print(col.Fore.MAGENTA+'\nCalcul du RVI'+col.Style.RESET_ALL)
    def funny(df_all,_ticker,_window):
        _ticker = _ticker.replace('/','')
        globals()['df_%s' %_ticker] = pd.DataFrame()
        globals()['df_%s' %_ticker] = df_all[df_all.Symbol==_ticker]
        globals()['df_%s' %_ticker]['Std'] = globals()['df_%s' %_ticker].Close.rolling(window=_window).std()
        globals()['df_%s' %_ticker]['Positive'] = np.where((globals()['df_%s' %_ticker].Std > globals()['df_%s' %_ticker].Std.shift(1)),globals()['df_%s' %_ticker].Std,0)
        globals()['df_%s' %_ticker]['Negative'] = np.where((globals()['df_%s' %_ticker].Std < globals()['df_%s' %_ticker].Std.shift(1)),globals()['df_%s' %_ticker].Std,0)
        globals()['df_%s' %_ticker]['PoMA'] = Wilder(globals()['df_%s' %_ticker]['Positive'],_window)
        globals()['df_%s' %_ticker]['NeMA'] = Wilder(globals()['df_%s' %_ticker]['Negative'],_window)
        globals()['df_%s' %_ticker]['RVI'] = (100 * globals()['df_%s' %_ticker]['PoMA']) / (globals()['df_%s' %_ticker]['PoMA'] + globals()['df_%s' %_ticker]['NeMA'])
        globals()['df_%s' %_ticker] = globals()['df_%s' %_ticker].drop(['Std','Positive','Negative','PoMA','NeMA'],axis=1)
        return(globals()['df_%s' %_ticker])

    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(funny)(df_all=df_all,_ticker=_ticker,_window=_window) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    df_temp= df_temp.append(parallel_pool(delayed_funcs))
    return(df_temp.sort_index(axis=0))

def onlosma(df_all,TICKER_LIST,_window=8,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul ONLOSMA'+col.Style.RESET_ALL)

    '''df_all = La base à travailler, _fast = fenetre courte,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    print('On High Simple Moving Average Calculation')
    def on_lo(df_all,_window,_ticker):
        _ticker = _ticker.replace('/','')
        hourly = df_all[df_all.Symbol==_ticker].copy()
        hourly['ONLOSMA_'+str(_window)] = hourly.Low.rolling(_window).mean()
        return(hourly)
    tt = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(on_lo)(df_all=df_all,_window=_window,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
    
    
    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(tt[(tt.index<=end)&(tt.index>=start)&(tt.Symbol==_ticker.replace('/',''))].Close)
        ax0.plot(tt[(tt.index<=end)&(tt.index>=start)&(tt.Symbol==_ticker.replace('/',''))]['SMA_'+str(_window)])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close','SMA_'+str(_window)],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Simple Moving Average on Low Values from "+str(start)+' to '+str(end), fontsize = 20)
        plt.show()
    return(tt.sort_index(axis=0))

def onhisma(df_all,TICKER_LIST,_window=8,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul ONHISMA'+col.Style.RESET_ALL)
    '''df_all = La base à travailler, _fast = fenetre courte,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    print('On High Simple Moving Average Calculation')
    def on_hi(df_all,_window,_ticker):
        _ticker = _ticker.replace('/','')
        hourly = df_all[df_all.Symbol==_ticker].copy()
        hourly['ONHISMA_'+str(_window)] = hourly.High.rolling(_window).mean()
        return(hourly)
    tt = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(on_hi)(df_all=df_all,_window=_window,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
    
    
    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(tt[(tt.index<=end)&(tt.index>=start)&(tt.Symbol==_ticker.replace('/',''))].Close)
        ax0.plot(tt[(tt.index<=end)&(tt.index>=start)&(tt.Symbol==_ticker.replace('/',''))]['SMA_'+str(_window)])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['Close','SMA_'+str(_window)],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Simple Moving Average on High Values from "+str(start)+' to '+str(end), fontsize = 20)
        plt.show()
    return(tt.sort_index(axis=0))

def atr(df_all,TICKER_LIST,_window=14,_plot=0,_ticker=None,start=None,end=None):
    print(col.Fore.MAGENTA+'\nCalcul ATR'+col.Style.RESET_ALL)
    '''df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
    _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

    ##### On relève le close de la bougie précédente
    df_all['prev_close'] = df_all.groupby('Symbol')['Close'].shift(1)

    ##### On récupère le maximum parmi 3 possibilités :
        ##### High - Low
        ##### High moins close précédent
        ##### Close précédent - Low
    df_all['TR'] = np.maximum((df_all['High'] - df_all['Low']), 
                        np.maximum(abs(df_all['High'] - df_all['prev_close']), 
                        abs(df_all['prev_close'] - df_all['Low'])))
    
    def get_atr(df_all,_ticker):
        
        _ticker = _ticker.replace('/','') 
        df = df_all[df_all.Symbol == _ticker].copy()
        print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,_ticker,col.Style.RESET_ALL,end='',flush=True)
        df.loc[df.Symbol==_ticker,'ATR_'+str(_window)] = Wilder(df['TR'], _window)
        return(df.sort_index(axis=0))

    tt = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(get_atr)(df_all=df_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
        
    tt = tt.drop(['prev_close','TR'],axis=1)

    if _plot == 1:
        sns.set()
        fig = plt.figure(facecolor = 'white', figsize = (30,5))
        ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
        ax0.plot(tt[(tt.index<=end)&(tt.index>=start)&(tt.Symbol==_ticker.replace('/',''))]['ATR_'+str(_window)])
        ax0.set_facecolor('ghostwhite')
        ax0.legend(['ATR_'+str(_window)],ncol=3, loc = 'upper left', fontsize = 15)
        plt.title(_ticker+" Average True Range from "+str(start)+' to '+str(end), fontsize = 20)

    return(tt.sort_index(axis=0))

def pivot(weekly_all, TICKER_LIST):
    print(col.Fore.MAGENTA+'\nCalcul des PIVOT, RESISTANCE ET SUPPORT'+col.Style.RESET_ALL)
    def founk(weekly_all,_ticker):
        _ticker = _ticker.replace('/','')
        weekly_temp = weekly_all[weekly_all.Symbol==_ticker]
        weekly_temp['S38'] = weekly_temp.PP - (0.382 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['S62'] = weekly_temp.PP - (0.618 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['S100'] = weekly_temp.PP - (1 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['S138'] = weekly_temp.PP - (1.382 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['S162'] = weekly_temp.PP - (1.618 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['S200'] = weekly_temp.PP - (2 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['R38'] = weekly_temp.PP + (0.382 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['R62'] = weekly_temp.PP + (0.618 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['R100'] = weekly_temp.PP + (1 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['R138'] = weekly_temp.PP + (1.382 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['R162'] = weekly_temp.PP + (1.618 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        weekly_temp['R200'] = weekly_temp.PP + (2 * (weekly_temp.High.shift(1) - weekly_temp.Low.shift(1)))
        return(weekly_temp)
    df = pd.DataFrame()
    weekly_all['PP'] = (weekly_all.groupby('Symbol').High.shift(1) + weekly_all.groupby('Symbol').Low.shift(1) + weekly_all.groupby('Symbol').Close.shift(1)) / 3
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(founk)(weekly_all=weekly_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    df= df.append(parallel_pool(delayed_funcs))
    return(df.sort_index(axis=0))

def pivotimportdaily(daily_all,weekly_all,TICKER_LIST):
    print(col.Fore.MAGENTA+'\nCalcul du PIVOT IMPORT'+col.Style.RESET_ALL)
    PP = []
    S38 = []
    S62 = []
    S100 = []
    S138 = []
    S162 = []
    S200 = []
    R38 = []
    R62 = []
    R100 = []
    R138 = []
    R162 = []
    R200 = []

    def foolie(daily_all,weekly_all,_ticker):
        _ticker = _ticker.replace('/','')
        globals()['weekly_temp_%s' %_ticker]= weekly_all[weekly_all.Symbol==_ticker].copy()
        globals()['daily_temp_%s' %_ticker]= daily_all[daily_all.Symbol==_ticker].copy()
        for i in range(len(globals()['daily_temp_%s' %_ticker])):
            PP.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].PP[0])
            S38.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].S38[0])
            S62.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].S62[0])
            S100.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].S100[0])
            S138.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].S138[0])
            S162.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].S162[0])
            S200.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].S200[0])
            R38.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].R38[0])
            R62.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].R62[0])
            R100.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].R100[0])
            R138.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].R138[0])
            R162.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].R162[0])
            R200.append(globals()['weekly_temp_%s' %_ticker][(globals()['weekly_temp_%s' %_ticker].Date.dt.week == globals()['daily_temp_%s' %_ticker].Date.dt.week[i])&\
                    (globals()['weekly_temp_%s' %_ticker].Date.dt.year == globals()['daily_temp_%s' %_ticker].Date.dt.year[i])].R200[0])
        globals()['daily_temp_%s' %_ticker]['PP'] = PP
        globals()['daily_temp_%s' %_ticker]['S38'] = S38
        globals()['daily_temp_%s' %_ticker]['S62'] = S62
        globals()['daily_temp_%s' %_ticker]['S100'] = S100
        globals()['daily_temp_%s' %_ticker]['S138'] = S138
        globals()['daily_temp_%s' %_ticker]['S162'] = S162
        globals()['daily_temp_%s' %_ticker]['S200'] = S200
        globals()['daily_temp_%s' %_ticker]['R38'] = R38
        globals()['daily_temp_%s' %_ticker]['R62'] = R62
        globals()['daily_temp_%s' %_ticker]['R100'] = R100
        globals()['daily_temp_%s' %_ticker]['R138'] = R138
        globals()['daily_temp_%s' %_ticker]['R162'] = R162
        globals()['daily_temp_%s' %_ticker]['R200'] = R200
        return(globals()['daily_temp_%s' %_ticker].sort_index(axis=0))
    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(foolie)(daily_all=daily_all,weekly_all=weekly_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes")
    df_temp = df_temp.append(parallel_pool(delayed_funcs))
    return(df_temp.sort_index(axis=0))

def pivotimportdf(df_all,weekly_all,TICKER_LIST):
    def get_ppnco(weekly_all,df_all,_ticker):
        _ticker = _ticker.replace('/','')
        weekly = weekly_all[weekly_all.Symbol==_ticker].copy()
        hourly = df_all[df_all.Symbol==_ticker].copy()
        weekly['Date'] = pd.to_datetime(weekly.Date)
        hourly['Date'] = pd.to_datetime(hourly.Date)
        hourly = hourly.join(weekly[['PP','S38','S62','S100','S138','S162','S200','R38','R62','R100','R138','R162','R200','Date']],how='left',on='Date',rsuffix='_2drop')
        hourly = hourly.drop(['Date_2drop'],axis=1)
        return(hourly.sort_index(axis=0))
        
    tt = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(get_ppnco)(weekly_all=weekly_all,df_all=df_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
    tt.PP.fillna(method='ffill', inplace=True)
    tt.S38.fillna(method='ffill', inplace=True)
    tt.S62.fillna(method='ffill', inplace=True)
    tt.S100.fillna(method='ffill', inplace=True)
    tt.S138.fillna(method='ffill', inplace=True)
    tt.S162.fillna(method='ffill', inplace=True)
    tt.S200.fillna(method='ffill', inplace=True)
    tt.R38.fillna(method='ffill', inplace=True)
    tt.R62.fillna(method='ffill', inplace=True)
    tt.R100.fillna(method='ffill', inplace=True)
    tt.R138.fillna(method='ffill', inplace=True)
    tt.R162.fillna(method='ffill', inplace=True)
    tt.R200.fillna(method='ffill', inplace=True)
    return(tt.sort_index(axis=0))

def adr(daily_all,_window):
    print(col.Fore.MAGENTA+'\nCalcul du ADR'+col.Style.RESET_ALL)
    temp = pd.DataFrame()
    for _ticker in tqdm(daily_all.Symbol.unique()):
        daily = daily_all[daily_all.Symbol==_ticker].copy()
        daily['ADR'] = (daily.High - daily.Low).rolling(_window).mean().shift(1)
        temp = temp.append(daily)
        temp = temp.drop(['list','Week','WeekDay','WeekNo','Year'],axis=1)
    return(temp.sort_index(axis=0))

def getadr(daily_all,df_all, TICKER_LIST):
    print("\nRécupération de l'ADR en cours...")
    def get_ohlc(df_all,other_all,_ticker,_suffix='_2Drop'):
        _ticker = _ticker.replace('/','')
        other = other_all[other_all.Symbol==_ticker].copy()
        hourly = df_all[df_all.Symbol==_ticker].copy()
        other['Date'] = pd.to_datetime(other.Date)
        hourly['Date'] = pd.to_datetime(hourly.Date)
        hourly = hourly.join(other[['ADR']],how='left',on='Date',rsuffix=_suffix)
        hourly = hourly.join(other[['High']],how='left',on='Date',rsuffix=_suffix)
        hourly = hourly.join(other[['Low']],how='left',on='Date',rsuffix=_suffix)
        hourly = hourly.rename(columns={'High'+_suffix: "DayHigh", 'Low'+_suffix: "DayLow"})
        try:
            hourly = hourly.drop(['Date'+_suffix],axis=1)
        except:
            pass
        return(hourly.sort_index(axis=0))
        
    tt = pd.DataFrame()
    _suffix = '_2Drop'
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(get_ohlc)(df_all=df_all,other_all=daily_all,_ticker=_ticker,_suffix=_suffix) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
    tt['ADR'].fillna(method='ffill', inplace=True)

    engine.say("Bim, the job is done")
    engine.runAndWait()
    return(tt.sort_index(axis=0))

def adrhnl(daily_all,df_all,TICKER_LIST):
    print(col.Fore.CYAN+'\nCalcul du ADR High & Low'+col.Style.RESET_ALL)
    print('En cours...')
    engine.say("Processing A D R Strategy High and Low")
    engine.runAndWait()
    
    def fh(row):
        global _flagh, val
        if row['Date'] != row['DateShiftMinus']: # pd.isnull(row['DateShiftMinus']) == True or 
            val = row['High']
            _flagl == 0
        if row['Date'] != row['DateShiftPlus']: # pd.isnull(row['DateShiftPlus']) == True or 
            val = row['DayHigh']
            _flagh=0
        elif row['High'] < row['DayHigh'] and row['High'] < row['HighShift'] and _flagh == 0 and row['Date'] == row['DateShiftMinus']:
            val = np.nan
            _flagh = 0
        elif row['High'] < row['DayHigh'] and row['High'] >= row['HighShift'] and _flagh == 0:
            val = row['High']
            _flagh = 0
        elif row['High'] == row['DayHigh'] and _flagh == 0:
            _flagh = 1
            val = row['DayHigh']
        elif _flagh == 1:
            val = row['DayHigh']          
        return(val)

    def fl(row):
        global _flagl, val
        if row['Date'] != row['DateShiftMinus']: # pd.isnull(row['DateShiftPlus']) == True or 
            val = row['Low']
            _flagl == 0   
        if row['Date'] != row['DateShiftPlus']: # pd.isnull(row['DateShiftMinus']) == True or 
            val = row['DayLow']
            _flagl = 0
        elif row['Low'] > row['DayLow'] and row['Low'] > row['LowShift'] and _flagl == 0 and row['Date'] == row['DateShiftMinus']:
            _flagl = 0
            val = np.nan
        elif row['Low'] > row['DayLow'] and row['Low'] <= row['LowShift'] and _flagl == 0:
            val = row['Low']
            _flagl = 0
        elif row['Low'] == row['DayLow'] and row['Low'] < row['LowShift']  and _flagl == 0:
            _flagl = 1
            val = row['DayLow']
        elif _flagl == 1:
            val = row['DayLow']
        return(val)

    
    def bidiouk(daily_all,df_all,_ticker):
        global _flagl, val,_flagh
        _flagh = 0
        _flagl = 0
        val = 0
        _ticker = _ticker.replace('/','')
        daily = daily_all[daily_all.Symbol==_ticker].copy()
        hourly = df_all[df_all.Symbol==_ticker].copy()
        
        hourly['DateShiftMinus'] = hourly.Date.shift(1)
        hourly['DateShiftPlus'] = hourly.Date.shift(-1)

        hourly['HighShift'] = hourly.High.shift(1)
        hourly['LowShift'] = hourly.Low.shift(1)

        hourly['HighSlope'] = hourly.apply(fh,axis=1)
        hourly['LowSlope'] = hourly.apply(fl,axis=1)
        hourly['HighSlope'].fillna(method='ffill', inplace=True)
        hourly['LowSlope'].fillna(method='ffill', inplace=True)

        hourly['ADR_High'] = hourly.LowSlope + hourly.ADR
        hourly['ADR_Low'] = hourly.HighSlope - hourly.ADR
        return(hourly.sort_index(axis=0))
    
    df_temp = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(bidiouk)(daily_all=daily_all,df_all=df_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    df_temp = df_temp.append(parallel_pool(delayed_funcs))
    df_temp = df_temp.drop(['DateShiftMinus','DateShiftPlus','HighShift','LowShift','HighSlope','LowSlope'],axis=1)
    engine.say("Bim, the job is done")
    engine.runAndWait()
    return(df_temp.sort_index(axis=0))

def sbgamma(df_all,TICKER_LIST):
    def sb_gam(df_all,_ticker):
        _ticker = _ticker.replace('/','')
        hourly = df_all[df_all.Symbol==_ticker].copy()
        _op1 = (hourly.Close - hourly.Open)/(hourly.Close.shift(1) - hourly.Open.shift(1))
        _op2 = (hourly.Close - hourly.Open)/(hourly.CloseAsk.shift(1) - hourly.OpenAsk.shift(1))
        _op3 = (hourly.Close - hourly.Open)/(hourly.CloseBid.shift(1) - hourly.OpenBid.shift(1))
        _op4 = (hourly.Close - hourly.Open)/(hourly.CloseBid.shift(1) - hourly.OpenAsk.shift(1))
        _op5 = (hourly.Close - hourly.Open)/(hourly.CloseAsk.shift(1) - hourly.OpenBid.shift(1))

        _condition1 = hourly.Close.shift(1) != hourly.Open.shift(1)
        _condition2 = hourly.CloseAsk.shift(1) != hourly.OpenAsk.shift(1)
        _condition3 = hourly.CloseBid.shift(1) != hourly.OpenBid.shift(1)
        _condition4 = hourly.CloseBid.shift(1) != hourly.OpenAsk.shift(1)
        _condition5 = hourly.CloseAsk.shift(1) != hourly.OpenBid.shift(1)
            
        hourly['SB_Gamma'] = np.where(_condition1,_op1,np.where(_condition2,_op2,np.where(_condition3,_op3,np.where(_condition4,_op4,np.where(_condition5,_op5,1.93E13)))))
        return(hourly)
    tt = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(sb_gam)(df_all=df_all,_ticker=_ticker) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
    return(tt.sort_index(axis=0))

def importohlc(df_all,other_all,TICKER_LIST,_suffix):
    print('Récupération des OHLC en cours...')
    def get_ohlc(df_all,other_all,_ticker,_suffix):
        _ticker = _ticker.replace('/','')
        other = other_all[other_all.Symbol==_ticker].copy()
        hourly = df_all[df_all.Symbol==_ticker].copy()
        other['Date'] = pd.to_datetime(other.Date)
        hourly['Date'] = pd.to_datetime(hourly.Date)
        hourly = hourly.join(other[['Open','High','Low','Close']],how='left',on='Date',rsuffix=_suffix)
        try:
            hourly = hourly.drop(['Date'+_suffix],axis=1)
        except:
            pass
        return(hourly.sort_index(axis=0))
        
    tt = pd.DataFrame()
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(get_ohlc)(df_all=df_all,other_all=other_all,_ticker=_ticker,_suffix=_suffix) for _ticker in tqdm(TICKER_LIST)]
    parallel_pool = Parallel(n_jobs=number_of_cpu,prefer="processes") # number_of_cpu
    tt = tt.append(parallel_pool(delayed_funcs))
    tt['Open'+_suffix].fillna(method='ffill', inplace=True)
    tt['High'+_suffix].fillna(method='ffill', inplace=True)
    tt['Low'+_suffix].fillna(method='ffill', inplace=True)
    tt['Close'+_suffix].fillna(method='ffill', inplace=True)
    return(tt.sort_index(axis=0))

def featuring(df_all):
    """[Entrer la df préparée avec les bons indictaurs. Renvoie une nouvelle df qu'avec les features + Symbol + Date + Signal]

    Args:
        df_all ([dataframe]): [Mettre la df qui doit être featurée.]
    """    
    features = pd.DataFrame(index=df_all.index)
    features['Symbol'] = df_all['Symbol']
    features['Date'] = df_all['Date']
    features['FEMA_21'] = df_all['Close'] - df_all['EMA_21']
    features['FEMA_8'] = df_all['Close'] - df_all['EMA_8']
    features['FADRLo'] = df_all['Close'] - df_all['ADR_Low']
    features['FADRHi'] = df_all['Close'] - df_all['ADR_High']
    features['FRVI40'] = df_all['RVI'] - 40
    features['FRVI60'] = df_all['RVI'] - 60
    features['FONLOSMA5'] = df_all['Low'] - df_all['ONLOSMA_5']
    features['FONHISMA5'] = df_all['High'] - df_all['ONHISMA_5']
    features['FONLOSMA21'] = df_all['Low'] - df_all['ONLOSMA_21']
    features['FONHISMA21'] = df_all['High'] - df_all['ONHISMA_21']
    features['FONLOSMA34'] = df_all['Low'] - df_all['ONLOSMA_34']
    features['FONHISMA34'] = df_all['High'] - df_all['ONHISMA_34']
    features['FSBGAMMA'] = df_all['SB_Gamma']
    features['FOPENWEEKLY'] = df_all['Close'] - df_all['Open_weekly'].shift(1)
    features['FHIGHWEEKLY'] = df_all['Close'] - df_all['High_weekly'].shift(1)
    features['FLOWWEEKLY'] = df_all['Close'] - df_all['Low_weekly'].shift(1)
    features['FCLOSEWEEKLY'] = df_all['Close'] - df_all['Close_weekly'].shift(1)
    features['FOPENDAILY'] = df_all['Close'] - df_all['Open_daily'].shift(1)
    features['FHIGHDAILY'] = df_all['Close'] - df_all['High_daily'].shift(1)
    features['FLOWDAILY'] = df_all['Close'] - df_all['Low_daily'].shift(1)
    features['FCLOSEDAILY'] = df_all['Close'] - df_all['Close_daily'].shift(1)
    features['FOPENHOURLY'] = df_all['Close'] - df_all['Open_daily'].shift(1)
    features['FHIGHHOURLY'] = df_all['Close'] - df_all['High_daily'].shift(1)
    features['FLOWHOURLY'] = df_all['Close'] - df_all['Low_daily'].shift(1)
    features['FCLOSEHOURLY'] = df_all['Close'] - df_all['Close_daily'].shift(1)
    features['FSMA200'] = df_all['Close'] - df_all['SMA_200']
    features['FBOLUP20'] = df_all['Close'] - df_all['UpperBand']
    features['FBOLLOW20'] = df_all['Close'] - df_all['LowerBand']
    features['FPP'] = df_all['Close'] - df_all['PP']
    features['FS38'] = df_all['Close'] - df_all['S38']
    features['FS62'] = df_all['Close'] - df_all['S62']
    features['FS100'] = df_all['Close'] - df_all['S100']
    features['FS138'] = df_all['Close'] - df_all['S138']
    features['FS162'] = df_all['Close'] - df_all['S162']
    features['FS200'] = df_all['Close'] - df_all['S200']
    features['FR38'] = df_all['Close'] - df_all['R38']
    features['FR62'] = df_all['Close'] - df_all['R62']
    features['FR100'] = df_all['Close'] - df_all['R100']
    features['FR138'] = df_all['Close'] - df_all['R138']
    features['FR162'] = df_all['Close'] - df_all['R162']
    features['FR200'] = df_all['Close'] - df_all['R200']
    features['SBATR'] = (df_all['Close'] - df_all['Open']) / df_all['ATR_14']
    features['Signal'] = df_all['Signal']
    return(features)

def scaling(features,_ticker,TRACKER,_start,_mid,_stop,_last,scaler):
    """[Entrer la df deja featuree pour effectuer dessus le scaling MinMax. Renvoie la df actuelle => possibilité d'écraser]

    Args:
        features ([DataFrame]): [La dataframe qui a déjà été featurée]
        _ticker ([String]): [Le ticker sans le slash. De type EURUSD]
        TRACKER ([String]): [Liste des tickers avec slash. De type EUR/USD]
        _start ([Date]): [Date de départ du train]
        _mid ([Date]): [Date du split entre train et test]
        _stop ([Date]): [Date de fin du test et donc du début du OOS]
        scaler ([type]): [description]
    """    
    features = features.dropna()
    features = features[features.Symbol==_ticker]
    features['TRACKER'] = np.where(features.index.isin(TRACKER),1,0)
    features_train = features[(features.Date>=_start)&(features.Date<=_mid)]
    features_test = features[(features.Date>_mid)&(features.Date<=_stop)]
    features_oos = features[(features.Date>_stop)&(features.Date <= _last)]
    features_train = features_train[features_train.Signal!=0]
    features_test = features_test[features_test.Signal!=0]

    features_train['FEMA_21'] = scaler.fit_transform(np.nan_to_num(features_train.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_train['FEMA_8'] = scaler.fit_transform(np.nan_to_num(features_train.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_train['FADRLo'] = scaler.fit_transform(np.nan_to_num(features_train.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_train['FADRHi'] = scaler.fit_transform(np.nan_to_num(features_train.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_train['FRVI40'] = scaler.fit_transform(np.nan_to_num(features_train.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_train['FRVI60'] = scaler.fit_transform(np.nan_to_num(features_train.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA5'] = scaler.fit_transform(np.nan_to_num(features_train.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_train['FONHISMA5'] = scaler.fit_transform(np.nan_to_num(features_train.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA21'] = scaler.fit_transform(np.nan_to_num(features_train.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_train['FONHISMA21'] = scaler.fit_transform(np.nan_to_num(features_train.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA34'] = scaler.fit_transform(np.nan_to_num(features_train.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_train['FSBGAMMA'] = scaler.fit_transform(np.nan_to_num(features_train.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_train.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_train.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_train.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_train.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENDAILY'] = scaler.fit_transform(np.nan_to_num(features_train.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHDAILY'] = scaler.fit_transform(np.nan_to_num(features_train.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWDAILY'] = scaler.fit_transform(np.nan_to_num(features_train.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEDAILY'] = scaler.fit_transform(np.nan_to_num(features_train.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENHOURLY'] = scaler.fit_transform(np.nan_to_num(features_train.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHHOURLY'] = scaler.fit_transform(np.nan_to_num(features_train.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWHOURLY'] = scaler.fit_transform(np.nan_to_num(features_train.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEHOURLY'] = scaler.fit_transform(np.nan_to_num(features_train.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FSMA200'] = scaler.fit_transform(np.nan_to_num(features_train.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_train['FBOLUP20'] = scaler.fit_transform(np.nan_to_num(features_train.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_train['FPP'] = scaler.fit_transform(np.nan_to_num(features_train.FPP.astype(np.float32)).reshape(-1, 1))
    features_train['FS38'] = scaler.fit_transform(np.nan_to_num(features_train.FS38.astype(np.float32)).reshape(-1, 1))
    features_train['FS62'] = scaler.fit_transform(np.nan_to_num(features_train.FS62.astype(np.float32)).reshape(-1, 1))
    features_train['FS100'] = scaler.fit_transform(np.nan_to_num(features_train.FS100.astype(np.float32)).reshape(-1, 1))
    features_train['FS138'] = scaler.fit_transform(np.nan_to_num(features_train.FS138.astype(np.float32)).reshape(-1, 1))
    features_train['FR162'] = scaler.fit_transform(np.nan_to_num(features_train.FS162.astype(np.float32)).reshape(-1, 1))
    features_train['FS200'] = scaler.fit_transform(np.nan_to_num(features_train.FS200.astype(np.float32)).reshape(-1, 1))
    features_train['FR38'] = scaler.fit_transform(np.nan_to_num(features_train.FR38.astype(np.float32)).reshape(-1, 1))
    features_train['FR62'] = scaler.fit_transform(np.nan_to_num(features_train.FR62.astype(np.float32)).reshape(-1, 1))
    features_train['FR100'] = scaler.fit_transform(np.nan_to_num(features_train.FR100.astype(np.float32)).reshape(-1, 1))
    features_train['FR138'] = scaler.fit_transform(np.nan_to_num(features_train.FR138.astype(np.float32)).reshape(-1, 1))
    features_train['FR162'] = scaler.fit_transform(np.nan_to_num(features_train.FR162.astype(np.float32)).reshape(-1, 1))
    features_train['FR200'] = scaler.fit_transform(np.nan_to_num(features_train.FR200.astype(np.float32)).reshape(-1, 1))
    features_train['SBATR'] = scaler.fit_transform(np.nan_to_num(features_train.SBATR.astype(np.float32)).reshape(-1, 1))
    
    features_test['FEMA_21'] = scaler.fit_transform(np.nan_to_num(features_test.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_test['FEMA_8'] = scaler.fit_transform(np.nan_to_num(features_test.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_test['FADRLo'] = scaler.fit_transform(np.nan_to_num(features_test.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_test['FADRHi'] = scaler.fit_transform(np.nan_to_num(features_test.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_test['FRVI40'] = scaler.fit_transform(np.nan_to_num(features_test.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_test['FRVI60'] = scaler.fit_transform(np.nan_to_num(features_test.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA5'] = scaler.fit_transform(np.nan_to_num(features_test.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_test['FONHISMA5'] = scaler.fit_transform(np.nan_to_num(features_test.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA21'] = scaler.fit_transform(np.nan_to_num(features_test.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_test['FONHISMA21'] = scaler.fit_transform(np.nan_to_num(features_test.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA34'] = scaler.fit_transform(np.nan_to_num(features_test.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_test['FSBGAMMA'] = scaler.fit_transform(np.nan_to_num(features_test.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_test.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_test.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_test.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_test.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENDAILY'] = scaler.fit_transform(np.nan_to_num(features_test.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHDAILY'] = scaler.fit_transform(np.nan_to_num(features_test.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWDAILY'] = scaler.fit_transform(np.nan_to_num(features_test.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEDAILY'] = scaler.fit_transform(np.nan_to_num(features_test.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENHOURLY'] = scaler.fit_transform(np.nan_to_num(features_test.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHHOURLY'] = scaler.fit_transform(np.nan_to_num(features_test.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWHOURLY'] = scaler.fit_transform(np.nan_to_num(features_test.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEHOURLY'] = scaler.fit_transform(np.nan_to_num(features_test.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FSMA200'] = scaler.fit_transform(np.nan_to_num(features_test.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_test['FBOLUP20'] = scaler.fit_transform(np.nan_to_num(features_test.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_test['FPP'] = scaler.fit_transform(np.nan_to_num(features_test.FPP.astype(np.float32)).reshape(-1, 1))
    features_test['FS38'] = scaler.fit_transform(np.nan_to_num(features_test.FS38.astype(np.float32)).reshape(-1, 1))
    features_test['FS62'] = scaler.fit_transform(np.nan_to_num(features_test.FS62.astype(np.float32)).reshape(-1, 1))
    features_test['FS100'] = scaler.fit_transform(np.nan_to_num(features_test.FS100.astype(np.float32)).reshape(-1, 1))
    features_test['FS138'] = scaler.fit_transform(np.nan_to_num(features_test.FS138.astype(np.float32)).reshape(-1, 1))
    features_test['FR162'] = scaler.fit_transform(np.nan_to_num(features_test.FS162.astype(np.float32)).reshape(-1, 1))
    features_test['FS200'] = scaler.fit_transform(np.nan_to_num(features_test.FS200.astype(np.float32)).reshape(-1, 1))
    features_test['FR38'] = scaler.fit_transform(np.nan_to_num(features_test.FR38.astype(np.float32)).reshape(-1, 1))
    features_test['FR62'] = scaler.fit_transform(np.nan_to_num(features_test.FR62.astype(np.float32)).reshape(-1, 1))
    features_test['FR100'] = scaler.fit_transform(np.nan_to_num(features_test.FR100.astype(np.float32)).reshape(-1, 1))
    features_test['FR138'] = scaler.fit_transform(np.nan_to_num(features_test.FR138.astype(np.float32)).reshape(-1, 1))
    features_test['FR162'] = scaler.fit_transform(np.nan_to_num(features_test.FR162.astype(np.float32)).reshape(-1, 1))
    features_test['FR200'] = scaler.fit_transform(np.nan_to_num(features_test.FR200.astype(np.float32)).reshape(-1, 1))
    features_test['SBATR'] = scaler.fit_transform(np.nan_to_num(features_test.SBATR.astype(np.float32)).reshape(-1, 1))

    features_oos['FEMA_21'] = scaler.fit_transform(np.nan_to_num(features_oos.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_oos['FEMA_8'] = scaler.fit_transform(np.nan_to_num(features_oos.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_oos['FADRLo'] = scaler.fit_transform(np.nan_to_num(features_oos.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_oos['FADRHi'] = scaler.fit_transform(np.nan_to_num(features_oos.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_oos['FRVI40'] = scaler.fit_transform(np.nan_to_num(features_oos.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_oos['FRVI60'] = scaler.fit_transform(np.nan_to_num(features_oos.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA5'] = scaler.fit_transform(np.nan_to_num(features_oos.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_oos['FONHISMA5'] = scaler.fit_transform(np.nan_to_num(features_oos.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA21'] = scaler.fit_transform(np.nan_to_num(features_oos.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_oos['FONHISMA21'] = scaler.fit_transform(np.nan_to_num(features_oos.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA34'] = scaler.fit_transform(np.nan_to_num(features_oos.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_oos['FSBGAMMA'] = scaler.fit_transform(np.nan_to_num(features_oos.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEWEEKLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENDAILY'] = scaler.fit_transform(np.nan_to_num(features_oos.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHDAILY'] = scaler.fit_transform(np.nan_to_num(features_oos.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWDAILY'] = scaler.fit_transform(np.nan_to_num(features_oos.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEDAILY'] = scaler.fit_transform(np.nan_to_num(features_oos.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENHOURLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHHOURLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWHOURLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEHOURLY'] = scaler.fit_transform(np.nan_to_num(features_oos.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FSMA200'] = scaler.fit_transform(np.nan_to_num(features_oos.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_oos['FBOLUP20'] = scaler.fit_transform(np.nan_to_num(features_oos.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_oos['FPP'] = scaler.fit_transform(np.nan_to_num(features_oos.FPP.astype(np.float32)).reshape(-1, 1))
    features_oos['FS38'] = scaler.fit_transform(np.nan_to_num(features_oos.FS38.astype(np.float32)).reshape(-1, 1))
    features_oos['FS62'] = scaler.fit_transform(np.nan_to_num(features_oos.FS62.astype(np.float32)).reshape(-1, 1))
    features_oos['FS100'] = scaler.fit_transform(np.nan_to_num(features_oos.FS100.astype(np.float32)).reshape(-1, 1))
    features_oos['FS138'] = scaler.fit_transform(np.nan_to_num(features_oos.FS138.astype(np.float32)).reshape(-1, 1))
    features_oos['FR162'] = scaler.fit_transform(np.nan_to_num(features_oos.FS162.astype(np.float32)).reshape(-1, 1))
    features_oos['FS200'] = scaler.fit_transform(np.nan_to_num(features_oos.FS200.astype(np.float32)).reshape(-1, 1))
    features_oos['FR38'] = scaler.fit_transform(np.nan_to_num(features_oos.FR38.astype(np.float32)).reshape(-1, 1))
    features_oos['FR62'] = scaler.fit_transform(np.nan_to_num(features_oos.FR62.astype(np.float32)).reshape(-1, 1))
    features_oos['FR100'] = scaler.fit_transform(np.nan_to_num(features_oos.FR100.astype(np.float32)).reshape(-1, 1))
    features_oos['FR138'] = scaler.fit_transform(np.nan_to_num(features_oos.FR138.astype(np.float32)).reshape(-1, 1))
    features_oos['FR162'] = scaler.fit_transform(np.nan_to_num(features_oos.FR162.astype(np.float32)).reshape(-1, 1))
    features_oos['FR200'] = scaler.fit_transform(np.nan_to_num(features_oos.FR200.astype(np.float32)).reshape(-1, 1))
    features_oos['SBATR'] = scaler.fit_transform(np.nan_to_num(features_oos.SBATR.astype(np.float32)).reshape(-1, 1))
    return(features_train,features_test,features_oos)

def quantile(features_train,features_test,features_oos,quantile_transform):
    """[Transformation par les Quantile]

    Args:
        features_train ([dataframe]): [train]
        features_test ([dataframe]): [test]
        features_oos ([dataframe]): [oos]
        quantile_transform ([sklearn]): [from preprocessing]
    """    
    
    features_train['FEMA_21'] = quantile_transform(np.nan_to_num(features_train.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_train['FEMA_8'] = quantile_transform(np.nan_to_num(features_train.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_train['FADRLo'] = quantile_transform(np.nan_to_num(features_train.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_train['FADRHi'] = quantile_transform(np.nan_to_num(features_train.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_train['FRVI40'] = quantile_transform(np.nan_to_num(features_train.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_train['FRVI60'] = quantile_transform(np.nan_to_num(features_train.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA5'] = quantile_transform(np.nan_to_num(features_train.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_train['FONHISMA5'] = quantile_transform(np.nan_to_num(features_train.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA21'] = quantile_transform(np.nan_to_num(features_train.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_train['FONHISMA21'] = quantile_transform(np.nan_to_num(features_train.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA34'] = quantile_transform(np.nan_to_num(features_train.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_train['FSBGAMMA'] = quantile_transform(np.nan_to_num(features_train.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENWEEKLY'] = quantile_transform(np.nan_to_num(features_train.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHWEEKLY'] = quantile_transform(np.nan_to_num(features_train.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWWEEKLY'] = quantile_transform(np.nan_to_num(features_train.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEWEEKLY'] = quantile_transform(np.nan_to_num(features_train.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENDAILY'] = quantile_transform(np.nan_to_num(features_train.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHDAILY'] = quantile_transform(np.nan_to_num(features_train.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWDAILY'] = quantile_transform(np.nan_to_num(features_train.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEDAILY'] = quantile_transform(np.nan_to_num(features_train.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENHOURLY'] = quantile_transform(np.nan_to_num(features_train.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHHOURLY'] = quantile_transform(np.nan_to_num(features_train.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWHOURLY'] = quantile_transform(np.nan_to_num(features_train.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEHOURLY'] = quantile_transform(np.nan_to_num(features_train.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FSMA200'] = quantile_transform(np.nan_to_num(features_train.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_train['FBOLUP20'] = quantile_transform(np.nan_to_num(features_train.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_train['FPP'] = quantile_transform(np.nan_to_num(features_train.FPP.astype(np.float32)).reshape(-1, 1))
    features_train['FS38'] = quantile_transform(np.nan_to_num(features_train.FS38.astype(np.float32)).reshape(-1, 1))
    features_train['FS62'] = quantile_transform(np.nan_to_num(features_train.FS62.astype(np.float32)).reshape(-1, 1))
    features_train['FS100'] = quantile_transform(np.nan_to_num(features_train.FS100.astype(np.float32)).reshape(-1, 1))
    features_train['FS138'] = quantile_transform(np.nan_to_num(features_train.FS138.astype(np.float32)).reshape(-1, 1))
    features_train['FR162'] = quantile_transform(np.nan_to_num(features_train.FS162.astype(np.float32)).reshape(-1, 1))
    features_train['FS200'] = quantile_transform(np.nan_to_num(features_train.FS200.astype(np.float32)).reshape(-1, 1))
    features_train['FR38'] = quantile_transform(np.nan_to_num(features_train.FR38.astype(np.float32)).reshape(-1, 1))
    features_train['FR62'] = quantile_transform(np.nan_to_num(features_train.FR62.astype(np.float32)).reshape(-1, 1))
    features_train['FR100'] = quantile_transform(np.nan_to_num(features_train.FR100.astype(np.float32)).reshape(-1, 1))
    features_train['FR138'] = quantile_transform(np.nan_to_num(features_train.FR138.astype(np.float32)).reshape(-1, 1))
    features_train['FR162'] = quantile_transform(np.nan_to_num(features_train.FR162.astype(np.float32)).reshape(-1, 1))
    features_train['FR200'] = quantile_transform(np.nan_to_num(features_train.FR200.astype(np.float32)).reshape(-1, 1))
    features_train['SBATR'] = quantile_transform(np.nan_to_num(features_train.SBATR.astype(np.float32)).reshape(-1, 1))
    
    features_test['FEMA_21'] = quantile_transform(np.nan_to_num(features_test.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_test['FEMA_8'] = quantile_transform(np.nan_to_num(features_test.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_test['FADRLo'] = quantile_transform(np.nan_to_num(features_test.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_test['FADRHi'] = quantile_transform(np.nan_to_num(features_test.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_test['FRVI40'] = quantile_transform(np.nan_to_num(features_test.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_test['FRVI60'] = quantile_transform(np.nan_to_num(features_test.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA5'] = quantile_transform(np.nan_to_num(features_test.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_test['FONHISMA5'] = quantile_transform(np.nan_to_num(features_test.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA21'] = quantile_transform(np.nan_to_num(features_test.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_test['FONHISMA21'] = quantile_transform(np.nan_to_num(features_test.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA34'] = quantile_transform(np.nan_to_num(features_test.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_test['FSBGAMMA'] = quantile_transform(np.nan_to_num(features_test.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENWEEKLY'] = quantile_transform(np.nan_to_num(features_test.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHWEEKLY'] = quantile_transform(np.nan_to_num(features_test.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWWEEKLY'] = quantile_transform(np.nan_to_num(features_test.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEWEEKLY'] = quantile_transform(np.nan_to_num(features_test.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENDAILY'] = quantile_transform(np.nan_to_num(features_test.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHDAILY'] = quantile_transform(np.nan_to_num(features_test.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWDAILY'] = quantile_transform(np.nan_to_num(features_test.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEDAILY'] = quantile_transform(np.nan_to_num(features_test.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENHOURLY'] = quantile_transform(np.nan_to_num(features_test.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHHOURLY'] = quantile_transform(np.nan_to_num(features_test.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWHOURLY'] = quantile_transform(np.nan_to_num(features_test.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEHOURLY'] = quantile_transform(np.nan_to_num(features_test.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FSMA200'] = quantile_transform(np.nan_to_num(features_test.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_test['FBOLUP20'] = quantile_transform(np.nan_to_num(features_test.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_test['FPP'] = quantile_transform(np.nan_to_num(features_test.FPP.astype(np.float32)).reshape(-1, 1))
    features_test['FS38'] = quantile_transform(np.nan_to_num(features_test.FS38.astype(np.float32)).reshape(-1, 1))
    features_test['FS62'] = quantile_transform(np.nan_to_num(features_test.FS62.astype(np.float32)).reshape(-1, 1))
    features_test['FS100'] = quantile_transform(np.nan_to_num(features_test.FS100.astype(np.float32)).reshape(-1, 1))
    features_test['FS138'] = quantile_transform(np.nan_to_num(features_test.FS138.astype(np.float32)).reshape(-1, 1))
    features_test['FR162'] = quantile_transform(np.nan_to_num(features_test.FS162.astype(np.float32)).reshape(-1, 1))
    features_test['FS200'] = quantile_transform(np.nan_to_num(features_test.FS200.astype(np.float32)).reshape(-1, 1))
    features_test['FR38'] = quantile_transform(np.nan_to_num(features_test.FR38.astype(np.float32)).reshape(-1, 1))
    features_test['FR62'] = quantile_transform(np.nan_to_num(features_test.FR62.astype(np.float32)).reshape(-1, 1))
    features_test['FR100'] = quantile_transform(np.nan_to_num(features_test.FR100.astype(np.float32)).reshape(-1, 1))
    features_test['FR138'] = quantile_transform(np.nan_to_num(features_test.FR138.astype(np.float32)).reshape(-1, 1))
    features_test['FR162'] = quantile_transform(np.nan_to_num(features_test.FR162.astype(np.float32)).reshape(-1, 1))
    features_test['FR200'] = quantile_transform(np.nan_to_num(features_test.FR200.astype(np.float32)).reshape(-1, 1))
    features_test['SBATR'] = quantile_transform(np.nan_to_num(features_test.SBATR.astype(np.float32)).reshape(-1, 1))

    features_oos['FEMA_21'] = quantile_transform(np.nan_to_num(features_oos.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_oos['FEMA_8'] = quantile_transform(np.nan_to_num(features_oos.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_oos['FADRLo'] = quantile_transform(np.nan_to_num(features_oos.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_oos['FADRHi'] = quantile_transform(np.nan_to_num(features_oos.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_oos['FRVI40'] = quantile_transform(np.nan_to_num(features_oos.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_oos['FRVI60'] = quantile_transform(np.nan_to_num(features_oos.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA5'] = quantile_transform(np.nan_to_num(features_oos.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_oos['FONHISMA5'] = quantile_transform(np.nan_to_num(features_oos.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA21'] = quantile_transform(np.nan_to_num(features_oos.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_oos['FONHISMA21'] = quantile_transform(np.nan_to_num(features_oos.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA34'] = quantile_transform(np.nan_to_num(features_oos.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_oos['FSBGAMMA'] = quantile_transform(np.nan_to_num(features_oos.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENWEEKLY'] = quantile_transform(np.nan_to_num(features_oos.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHWEEKLY'] = quantile_transform(np.nan_to_num(features_oos.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWWEEKLY'] = quantile_transform(np.nan_to_num(features_oos.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEWEEKLY'] = quantile_transform(np.nan_to_num(features_oos.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENDAILY'] = quantile_transform(np.nan_to_num(features_oos.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHDAILY'] = quantile_transform(np.nan_to_num(features_oos.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWDAILY'] = quantile_transform(np.nan_to_num(features_oos.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEDAILY'] = quantile_transform(np.nan_to_num(features_oos.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENHOURLY'] = quantile_transform(np.nan_to_num(features_oos.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHHOURLY'] = quantile_transform(np.nan_to_num(features_oos.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWHOURLY'] = quantile_transform(np.nan_to_num(features_oos.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEHOURLY'] = quantile_transform(np.nan_to_num(features_oos.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FSMA200'] = quantile_transform(np.nan_to_num(features_oos.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_oos['FBOLUP20'] = quantile_transform(np.nan_to_num(features_oos.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_oos['FPP'] = quantile_transform(np.nan_to_num(features_oos.FPP.astype(np.float32)).reshape(-1, 1))
    features_oos['FS38'] = quantile_transform(np.nan_to_num(features_oos.FS38.astype(np.float32)).reshape(-1, 1))
    features_oos['FS62'] = quantile_transform(np.nan_to_num(features_oos.FS62.astype(np.float32)).reshape(-1, 1))
    features_oos['FS100'] = quantile_transform(np.nan_to_num(features_oos.FS100.astype(np.float32)).reshape(-1, 1))
    features_oos['FS138'] = quantile_transform(np.nan_to_num(features_oos.FS138.astype(np.float32)).reshape(-1, 1))
    features_oos['FR162'] = quantile_transform(np.nan_to_num(features_oos.FS162.astype(np.float32)).reshape(-1, 1))
    features_oos['FS200'] = quantile_transform(np.nan_to_num(features_oos.FS200.astype(np.float32)).reshape(-1, 1))
    features_oos['FR38'] = quantile_transform(np.nan_to_num(features_oos.FR38.astype(np.float32)).reshape(-1, 1))
    features_oos['FR62'] = quantile_transform(np.nan_to_num(features_oos.FR62.astype(np.float32)).reshape(-1, 1))
    features_oos['FR100'] = quantile_transform(np.nan_to_num(features_oos.FR100.astype(np.float32)).reshape(-1, 1))
    features_oos['FR138'] = quantile_transform(np.nan_to_num(features_oos.FR138.astype(np.float32)).reshape(-1, 1))
    features_oos['FR162'] = quantile_transform(np.nan_to_num(features_oos.FR162.astype(np.float32)).reshape(-1, 1))
    features_oos['FR200'] = quantile_transform(np.nan_to_num(features_oos.FR200.astype(np.float32)).reshape(-1, 1))
    features_oos['SBATR'] = quantile_transform(np.nan_to_num(features_oos.SBATR.astype(np.float32)).reshape(-1, 1))

    return(features_train,features_test,features_oos)

def polytrans(features_train,features_test,features_oos,poly):
    """[Transformation par les Quantile]

    Args:
        features_train ([dataframe]): [train]
        features_test ([dataframe]): [test]
        features_oos ([dataframe]): [oos]
        poly.fit_transform ([sklearn]): [from preprocessing] 
    """    
    
    features_train['FEMA_21'] = poly.fit_transform(np.nan_to_num(features_train.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_train['FEMA_8'] = poly.fit_transform(np.nan_to_num(features_train.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_train['FADRLo'] = poly.fit_transform(np.nan_to_num(features_train.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_train['FADRHi'] = poly.fit_transform(np.nan_to_num(features_train.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_train['FRVI40'] = poly.fit_transform(np.nan_to_num(features_train.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_train['FRVI60'] = poly.fit_transform(np.nan_to_num(features_train.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA5'] = poly.fit_transform(np.nan_to_num(features_train.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_train['FONHISMA5'] = poly.fit_transform(np.nan_to_num(features_train.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA21'] = poly.fit_transform(np.nan_to_num(features_train.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_train['FONHISMA21'] = poly.fit_transform(np.nan_to_num(features_train.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_train['FONLOSMA34'] = poly.fit_transform(np.nan_to_num(features_train.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_train['FSBGAMMA'] = poly.fit_transform(np.nan_to_num(features_train.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENWEEKLY'] = poly.fit_transform(np.nan_to_num(features_train.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHWEEKLY'] = poly.fit_transform(np.nan_to_num(features_train.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWWEEKLY'] = poly.fit_transform(np.nan_to_num(features_train.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEWEEKLY'] = poly.fit_transform(np.nan_to_num(features_train.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENDAILY'] = poly.fit_transform(np.nan_to_num(features_train.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHDAILY'] = poly.fit_transform(np.nan_to_num(features_train.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWDAILY'] = poly.fit_transform(np.nan_to_num(features_train.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEDAILY'] = poly.fit_transform(np.nan_to_num(features_train.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_train['FOPENHOURLY'] = poly.fit_transform(np.nan_to_num(features_train.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FHIGHHOURLY'] = poly.fit_transform(np.nan_to_num(features_train.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FLOWHOURLY'] = poly.fit_transform(np.nan_to_num(features_train.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FCLOSEHOURLY'] = poly.fit_transform(np.nan_to_num(features_train.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_train['FSMA200'] = poly.fit_transform(np.nan_to_num(features_train.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_train['FBOLUP20'] = poly.fit_transform(np.nan_to_num(features_train.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_train['FPP'] = poly.fit_transform(np.nan_to_num(features_train.FPP.astype(np.float32)).reshape(-1, 1))
    features_train['FS38'] = poly.fit_transform(np.nan_to_num(features_train.FS38.astype(np.float32)).reshape(-1, 1))
    features_train['FS62'] = poly.fit_transform(np.nan_to_num(features_train.FS62.astype(np.float32)).reshape(-1, 1))
    features_train['FS100'] = poly.fit_transform(np.nan_to_num(features_train.FS100.astype(np.float32)).reshape(-1, 1))
    features_train['FS138'] = poly.fit_transform(np.nan_to_num(features_train.FS138.astype(np.float32)).reshape(-1, 1))
    features_train['FR162'] = poly.fit_transform(np.nan_to_num(features_train.FS162.astype(np.float32)).reshape(-1, 1))
    features_train['FS200'] = poly.fit_transform(np.nan_to_num(features_train.FS200.astype(np.float32)).reshape(-1, 1))
    features_train['FR38'] = poly.fit_transform(np.nan_to_num(features_train.FR38.astype(np.float32)).reshape(-1, 1))
    features_train['FR62'] = poly.fit_transform(np.nan_to_num(features_train.FR62.astype(np.float32)).reshape(-1, 1))
    features_train['FR100'] = poly.fit_transform(np.nan_to_num(features_train.FR100.astype(np.float32)).reshape(-1, 1))
    features_train['FR138'] = poly.fit_transform(np.nan_to_num(features_train.FR138.astype(np.float32)).reshape(-1, 1))
    features_train['FR162'] = poly.fit_transform(np.nan_to_num(features_train.FR162.astype(np.float32)).reshape(-1, 1))
    features_train['FR200'] = poly.fit_transform(np.nan_to_num(features_train.FR200.astype(np.float32)).reshape(-1, 1))
    features_train['SBATR'] = poly.fit_transform(np.nan_to_num(features_train.SBATR.astype(np.float32)).reshape(-1, 1))
    
    features_test['FEMA_21'] = poly.fit_transform(np.nan_to_num(features_test.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_test['FEMA_8'] = poly.fit_transform(np.nan_to_num(features_test.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_test['FADRLo'] = poly.fit_transform(np.nan_to_num(features_test.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_test['FADRHi'] = poly.fit_transform(np.nan_to_num(features_test.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_test['FRVI40'] = poly.fit_transform(np.nan_to_num(features_test.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_test['FRVI60'] = poly.fit_transform(np.nan_to_num(features_test.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA5'] = poly.fit_transform(np.nan_to_num(features_test.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_test['FONHISMA5'] = poly.fit_transform(np.nan_to_num(features_test.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA21'] = poly.fit_transform(np.nan_to_num(features_test.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_test['FONHISMA21'] = poly.fit_transform(np.nan_to_num(features_test.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_test['FONLOSMA34'] = poly.fit_transform(np.nan_to_num(features_test.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_test['FSBGAMMA'] = poly.fit_transform(np.nan_to_num(features_test.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENWEEKLY'] = poly.fit_transform(np.nan_to_num(features_test.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHWEEKLY'] = poly.fit_transform(np.nan_to_num(features_test.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWWEEKLY'] = poly.fit_transform(np.nan_to_num(features_test.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEWEEKLY'] = poly.fit_transform(np.nan_to_num(features_test.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENDAILY'] = poly.fit_transform(np.nan_to_num(features_test.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHDAILY'] = poly.fit_transform(np.nan_to_num(features_test.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWDAILY'] = poly.fit_transform(np.nan_to_num(features_test.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEDAILY'] = poly.fit_transform(np.nan_to_num(features_test.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_test['FOPENHOURLY'] = poly.fit_transform(np.nan_to_num(features_test.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FHIGHHOURLY'] = poly.fit_transform(np.nan_to_num(features_test.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FLOWHOURLY'] = poly.fit_transform(np.nan_to_num(features_test.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FCLOSEHOURLY'] = poly.fit_transform(np.nan_to_num(features_test.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_test['FSMA200'] = poly.fit_transform(np.nan_to_num(features_test.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_test['FBOLUP20'] = poly.fit_transform(np.nan_to_num(features_test.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_test['FPP'] = poly.fit_transform(np.nan_to_num(features_test.FPP.astype(np.float32)).reshape(-1, 1))
    features_test['FS38'] = poly.fit_transform(np.nan_to_num(features_test.FS38.astype(np.float32)).reshape(-1, 1))
    features_test['FS62'] = poly.fit_transform(np.nan_to_num(features_test.FS62.astype(np.float32)).reshape(-1, 1))
    features_test['FS100'] = poly.fit_transform(np.nan_to_num(features_test.FS100.astype(np.float32)).reshape(-1, 1))
    features_test['FS138'] = poly.fit_transform(np.nan_to_num(features_test.FS138.astype(np.float32)).reshape(-1, 1))
    features_test['FR162'] = poly.fit_transform(np.nan_to_num(features_test.FS162.astype(np.float32)).reshape(-1, 1))
    features_test['FS200'] = poly.fit_transform(np.nan_to_num(features_test.FS200.astype(np.float32)).reshape(-1, 1))
    features_test['FR38'] = poly.fit_transform(np.nan_to_num(features_test.FR38.astype(np.float32)).reshape(-1, 1))
    features_test['FR62'] = poly.fit_transform(np.nan_to_num(features_test.FR62.astype(np.float32)).reshape(-1, 1))
    features_test['FR100'] = poly.fit_transform(np.nan_to_num(features_test.FR100.astype(np.float32)).reshape(-1, 1))
    features_test['FR138'] = poly.fit_transform(np.nan_to_num(features_test.FR138.astype(np.float32)).reshape(-1, 1))
    features_test['FR162'] = poly.fit_transform(np.nan_to_num(features_test.FR162.astype(np.float32)).reshape(-1, 1))
    features_test['FR200'] = poly.fit_transform(np.nan_to_num(features_test.FR200.astype(np.float32)).reshape(-1, 1))
    features_test['SBATR'] = poly.fit_transform(np.nan_to_num(features_test.SBATR.astype(np.float32)).reshape(-1, 1))

    features_oos['FEMA_21'] = poly.fit_transform(np.nan_to_num(features_oos.FEMA_21.astype(np.float32)).reshape(-1, 1))
    features_oos['FEMA_8'] = poly.fit_transform(np.nan_to_num(features_oos.FEMA_8.astype(np.float32)).reshape(-1, 1))
    features_oos['FADRLo'] = poly.fit_transform(np.nan_to_num(features_oos.FADRLo.astype(np.float32)).reshape(-1, 1))
    features_oos['FADRHi'] = poly.fit_transform(np.nan_to_num(features_oos.FADRHi.astype(np.float32)).reshape(-1, 1))
    features_oos['FRVI40'] = poly.fit_transform(np.nan_to_num(features_oos.FRVI40.astype(np.float32)).reshape(-1, 1))
    features_oos['FRVI60'] = poly.fit_transform(np.nan_to_num(features_oos.FRVI60.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA5'] = poly.fit_transform(np.nan_to_num(features_oos.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
    features_oos['FONHISMA5'] = poly.fit_transform(np.nan_to_num(features_oos.FONHISMA5.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA21'] = poly.fit_transform(np.nan_to_num(features_oos.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
    features_oos['FONHISMA21'] = poly.fit_transform(np.nan_to_num(features_oos.FONHISMA21.astype(np.float32)).reshape(-1, 1))
    features_oos['FONLOSMA34'] = poly.fit_transform(np.nan_to_num(features_oos.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
    features_oos['FSBGAMMA'] = poly.fit_transform(np.nan_to_num(features_oos.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENWEEKLY'] = poly.fit_transform(np.nan_to_num(features_oos.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHWEEKLY'] = poly.fit_transform(np.nan_to_num(features_oos.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWWEEKLY'] = poly.fit_transform(np.nan_to_num(features_oos.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEWEEKLY'] = poly.fit_transform(np.nan_to_num(features_oos.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENDAILY'] = poly.fit_transform(np.nan_to_num(features_oos.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHDAILY'] = poly.fit_transform(np.nan_to_num(features_oos.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWDAILY'] = poly.fit_transform(np.nan_to_num(features_oos.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEDAILY'] = poly.fit_transform(np.nan_to_num(features_oos.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
    features_oos['FOPENHOURLY'] = poly.fit_transform(np.nan_to_num(features_oos.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FHIGHHOURLY'] = poly.fit_transform(np.nan_to_num(features_oos.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FLOWHOURLY'] = poly.fit_transform(np.nan_to_num(features_oos.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FCLOSEHOURLY'] = poly.fit_transform(np.nan_to_num(features_oos.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
    features_oos['FSMA200'] = poly.fit_transform(np.nan_to_num(features_oos.FSMA200.astype(np.float32)).reshape(-1, 1))
    features_oos['FBOLUP20'] = poly.fit_transform(np.nan_to_num(features_oos.FBOLUP20.astype(np.float32)).reshape(-1, 1))
    features_oos['FPP'] = poly.fit_transform(np.nan_to_num(features_oos.FPP.astype(np.float32)).reshape(-1, 1))
    features_oos['FS38'] = poly.fit_transform(np.nan_to_num(features_oos.FS38.astype(np.float32)).reshape(-1, 1))
    features_oos['FS62'] = poly.fit_transform(np.nan_to_num(features_oos.FS62.astype(np.float32)).reshape(-1, 1))
    features_oos['FS100'] = poly.fit_transform(np.nan_to_num(features_oos.FS100.astype(np.float32)).reshape(-1, 1))
    features_oos['FS138'] = poly.fit_transform(np.nan_to_num(features_oos.FS138.astype(np.float32)).reshape(-1, 1))
    features_oos['FR162'] = poly.fit_transform(np.nan_to_num(features_oos.FS162.astype(np.float32)).reshape(-1, 1))
    features_oos['FS200'] = poly.fit_transform(np.nan_to_num(features_oos.FS200.astype(np.float32)).reshape(-1, 1))
    features_oos['FR38'] = poly.fit_transform(np.nan_to_num(features_oos.FR38.astype(np.float32)).reshape(-1, 1))
    features_oos['FR62'] = poly.fit_transform(np.nan_to_num(features_oos.FR62.astype(np.float32)).reshape(-1, 1))
    features_oos['FR100'] = poly.fit_transform(np.nan_to_num(features_oos.FR100.astype(np.float32)).reshape(-1, 1))
    features_oos['FR138'] = poly.fit_transform(np.nan_to_num(features_oos.FR138.astype(np.float32)).reshape(-1, 1))
    features_oos['FR162'] = poly.fit_transform(np.nan_to_num(features_oos.FR162.astype(np.float32)).reshape(-1, 1))
    features_oos['FR200'] = poly.fit_transform(np.nan_to_num(features_oos.FR200.astype(np.float32)).reshape(-1, 1))
    features_oos['SBATR'] = poly.fit_transform(np.nan_to_num(features_oos.SBATR.astype(np.float32)).reshape(-1, 1))

    return(features_train,features_test,features_oos)

def check_inf(dataframe_to_check):
    df = dataframe_to_check.copy()
    # checking for infinity 
    print() 
    print("checking for infinity") 
    
    ds = df.isin([np.inf, -np.inf]) 
    #print(ds) 
    
    # printing the count of infinity values 
    print() 
    print("printing the count of infinity values") 
    
    count = np.isinf(df).values.sum() 
    print("It contains " + str(count) + " infinite values") 
    
    # counting infinity in a particular column name 
    c = np.isinf(df).values.sum() 
    print("It contains " + str(c) + " infinite values") 
    
    # printing column name where infinity is present 
    print() 
    print("printing column name where infinity is present") 
    col_name = df.columns.to_series()[np.isinf(df).any()] 
    print(col_name) 
    
    # printing row index with infinity 
    print() 
    print("printing len(r),len(ds),len(df)") 
    
    r = df.index[np.isinf(df).any(1)] 
    print(len(r),len(ds),len(df)) 

def check_nan(dataframe_to_check):
    df = dataframe_to_check.copy()
    # checking for infinity 
    print() 
    print("checking for NaN") 

    ds = df.isin([np.nan, -np.nan]) 
    #print(ds) 

    # printing the count of infinity values 
    print() 
    print("printing the count of NaN values") 

    count = np.isnan(df).values.sum() 
    print("It contains " + str(count) + " NaN values") 

    # counting infinity in a particular column name 
    c = np.isnan(df).values.sum() 
    print("It contains " + str(c) + " NaN values") 

    # printing column name where infinity is present 
    print() 
    print("printing column name where NaN is present") 
    col_name = df.columns.to_series()[np.isnan(df).any()] 
    print(col_name) 

    # printing row index with infinity 
    print() 
    print("printing len(r),len(ds),len(df)") 

    r = df.index[np.isnan(df).any(1)] 
    print(len(r),len(ds),len(df)) 

def is_we(dataframe_to_check):
    IDX = dataframe_to_check.index.to_list()
    c=0
    for day in tqdm(IDX):
        if day.weekday() == 5 or day.weekday() == 6:
            c += 1
    print('Nombre de samedi et dimanches présents :',c)

def idx_cross_df(dataframe_to_check,dataframe_reference):
    """[Check how much index of df1 are not in df2]

    Args:
        dataframe_to_check ([pandas dataframe]): [The dataframe in which we want to check if some index is missing]
        dataframe_reference ([pandas dataframe]): [The referencial dataframe which is supposed to have all the index]
    """    
    dataframe_to_check.loc[:,'datation'] = dataframe_to_check.index
    dataframe_reference.loc[:,'datation'] = dataframe_reference.index
    LIST = []
    REFERENCE = dataframe_reference.datation.to_list()
    for i in tqdm(range(len(dataframe_to_check))):
        if dataframe_to_check.datation[i] not in REFERENCE:
            LIST.append(dataframe_to_check.datation[i])
    print("\nNombre d'index de df à checker dans df de référence",len(LIST))

def missing_candle_hdd(dataframe_to_check,_period,_verbose=0):
    dataframe_to_check.index = pd.to_datetime(dataframe_to_check.Date+' '+dataframe_to_check.Time)
    dataframe_to_check['datation'] = dataframe_to_check.index
    if _period == 'm1':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=1)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'm5':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=5)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'm15':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=15)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'm30':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=30)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'H1':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(hours=1)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)

    dataframe_to_check['TimeJump'] = np.where(_condition,1,0)
    print("Nombre (shape) de bougies manquante comparé au nombre (shape) de bougie total",dataframe_to_check[dataframe_to_check.TimeJump==1].shape,dataframe_to_check.shape)
    print("Pourcentage de trous dans la base :",len(dataframe_to_check[dataframe_to_check.TimeJump==1])/len(dataframe_to_check)*100)
    if _verbose == 1:
        print(dataframe_to_check[dataframe_to_check.TimeJump==1])

def missing_candle(dataframe_to_check,_period,_verbose=0):
    dataframe_to_check['datation'] = dataframe_to_check.index
    if _period == 'm1':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=1)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'm5':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=5)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'm15':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=15)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'm30':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(minutes=30)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)
    elif _period == 'H1':
        _condition = (abs(dataframe_to_check.datation - dataframe_to_check.datation.shift(1)) != dt.timedelta(hours=1)) & (dataframe_to_check.datation.dt.day == dataframe_to_check.datation.shift(1).dt.day)

    dataframe_to_check['TimeJump'] = np.where(_condition,1,0)
    print("Nombre (shape) de bougies manquante comparé au nombre (shape) de bougie total",dataframe_to_check[dataframe_to_check.TimeJump==1].shape,dataframe_to_check.shape)
    print("Pourcentage de trous dans la base :",len(dataframe_to_check[dataframe_to_check.TimeJump==1])/len(dataframe_to_check)*100)
    if _verbose == 1:
        print(dataframe_to_check[dataframe_to_check.TimeJump==1])

def DagMaxBase(features_train):
    features_max = pd.DataFrame()
    for label in features_train.drop(['Date','Symbol'],axis=1).columns:
        features_max.loc[0,label] = abs(features[label]).max()   # max([abs(features_train[label]).max(),abs(features_train[label]).max()])
    joblib.dump(features_max,x.replace('/','')+'_MAX.dagmax')
    return(features_max)






if __name__ == "__main__":
    pass 