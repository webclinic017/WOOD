__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
####### RECUPERATION DONNEES ET PREPARATION DES DATA FX ############
####################################################################
####################################################################
        
import time     
import pandas as pd
import numpy as np
import colorama as col
from tqdm import tqdm
import joblib
#from joblib import Parallel,delayed
import datetime as dt
import fxcmpy
import pyttsx3
import datetime as dt
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,quantile_transform,PolynomialFeatures
engine = pyttsx3.init()
import subprocess, os, sys
print('version fxcmpy :',fxcmpy.__version__)

def golive(_period='m5',TICKER_LIST=['EUR/USD'],_target=0.002,_sl=0.001,_cash_ini=10,_token='dbdc379ce7761772c662c3e92250a0ae38385b2c',_server='demo'):
    _poz = 0
    x = TICKER_LIST[0]
    _ticker = x.replace('/','')
    TIK = ['AUD','NZD','GBP','JPY','CHF','CAD','SEK','NOK','ILS','MXN','USD','EUR']
    RATE = [0.776,0.721,1.3912,1/105.91,1/0.892,1/1.2681,1/8.2884,1/8.4261,1/3.2385,1/20.1564,1,1]
    df_ratefx = pd.DataFrame(index=TIK)
    df_ratefx['rate'] = RATE
    _scaler = MaxAbsScaler()
    
    def get_daily(df_all,TICKER_LIST):

        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        df_all = df_all[df_all.Symbol == _ticker]
        daily_all = pd.DataFrame(index=df_all.Date.unique())

        ##### Fabrication de la base daily
        daily_all['Lindex'] = list((df_all.groupby('Date').Date.first()))
        daily_all['Open'] = list((df_all.groupby('Date').Open.first()))
        daily_all['High'] = list((df_all.groupby('Date').High.max()))
        daily_all['Low'] = list((df_all.groupby('Date').Low.min()))
        daily_all['Close'] = list((df_all.groupby('Date').Close.last()))
        daily_all['Symbol'] = _ticker
        daily_all = daily_all.sort_values('Lindex') ##########
        daily_all.set_index(pd.to_datetime(daily_all.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True) #####
        daily_all['Date'] = daily_all.Lindex
        daily_all = daily_all.drop(['Lindex'],axis=1)
        
        #daily_all = daily_all.drop(['Lindex'],axis=1)
        return(daily_all.sort_index(axis=0))

    def get_weekly(daily_all,TICKER_LIST):

        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        daily_all = daily_all[daily_all.Symbol == _ticker]
        weekly_all = pd.DataFrame()
        weekly_all['Lindex'] = list((daily_all.groupby('Week').Date.first()))
        weekly_all['Open'] = list((daily_all.groupby('Week').Open.first()))
        weekly_all['High'] = list((daily_all.groupby('Week').High.max()))
        weekly_all['Low'] = list((daily_all.groupby('Week').Low.min()))
        weekly_all['Close'] = list((daily_all.groupby('Week').Close.last()))
        weekly_all = weekly_all.sort_values('Lindex')
        weekly_all.set_index(pd.to_datetime(weekly_all.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)
        weekly_all['Symbol'] = _ticker
        weekly_all['Date'] = weekly_all.Lindex
        weekly_all = weekly_all.drop(['Lindex'],axis=1)              
        #daily_all['WeekDay'] = np.where(daily_all.sort_values('Symbol').Week!=daily_all.sort_values('Symbol').Week.shift(1),daily_all.index,np.datetime64('NaT'))
        return(weekly_all.sort_index(axis=0))

    def timerange1D(df_all):
        # print('\nAjout Date')
        df_all['Date'] = df_all.index
        df_all['Date'] = df_all['Date'].dt.strftime(date_format='%Y-%m-%d')
        return(df_all.sort_index(axis=0))

    def timerange1W(daily_all):
        # print("\nAjout colonne 'Date dans le weekly" ) 
        daily_all['WeekNo'] = pd.to_datetime(daily_all.index)
        daily_all['WeekNo'] = daily_all['WeekNo'].dt.isocalendar().week.astype(str)
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
        # print(col.Fore.GREEN+"\nCalcul de l'EMA_"+str(_window)+"en cours"+col.Style.RESET_ALL)
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        df = df_all[df_all.Symbol==_ticker]
        df['EMA_'+str(_window)] = df.Close.ewm(span=_window,adjust=False).mean()
        return(df.sort_index(axis=0))

    def sma(df_all,_window=200):
        # print(col.Fore.MAGENTA+'\nCalcul SMA'+col.Style.RESET_ALL)
        '''Simple Moving Average (SMA)
        Simple Moving Average is one of the most common technical indicators. 
        SMA calculates the average of prices over a given interval of time and is used to determine the trend of the stock. 
        As defined above, I will create a slow SMA (SMA_15) and a fast SMA (SMA_5). 
        To provide Machine Learning algorithms with already engineered factors, 
        one can also use (SMA_15/SMA_5) or (SMA_15 - SMA_5) as a factor to capture the relationship between these two moving averages.
        df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''


        df_all['SMA_'+str(_window)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = _window).mean())
        return(df_all.sort_index(axis=0))

    def slowstochastic(df_all,TICKER_LIST,_window=5,_per=3):
        _ticker = TICKER_LIST[0]
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

    def bollinger(df_all,_slow=15):
        # print(col.Fore.MAGENTA+'\nCalcul BOLLINGER'+col.Style.RESET_ALL)
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

        return(df_all.sort_index(axis=0))

    def onlosma(df_all,TICKER_LIST,_window=8):
        # print(col.Fore.MAGENTA+'\nCalcul ONLOSMA'+col.Style.RESET_ALL)

        '''df_all = La base à travailler, _fast = fenetre courte,
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

        # print('On High Simple Moving Average Calculation')
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        hourly = df_all[df_all.Symbol==_ticker].copy()
        hourly['ONLOSMA_'+str(_window)] = hourly.Low.rolling(_window).mean()
        return(hourly.sort_index(axis=0))
        
    def onhisma(df_all,TICKER_LIST,_window=8):
        # print(col.Fore.MAGENTA+'\nCalcul ONHISMA'+col.Style.RESET_ALL)
        '''df_all = La base à travailler, _fast = fenetre courte,
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        hourly = df_all[df_all.Symbol==_ticker].copy()
        hourly['ONHISMA_'+str(_window)] = hourly.High.rolling(_window).mean()
        return(hourly.sort_index(axis=0))

    def atr(df_all,TICKER_LIST,_window=14):
        # print(col.Fore.MAGENTA+'\nCalcul ATR'+col.Style.RESET_ALL)
        '''df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
        _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''
        _ticker = TICKER_LIST[0]
        ##### On relève le close de la bougie précédente
        df_all['prev_close'] = df_all['Close'].shift(1)

        ##### On récupère le maximum parmi 3 possibilités :
            ##### High - Low
            ##### High moins close précédent
            ##### Close précédent - Low
        df_all['TR'] = np.maximum((df_all['High'] - df_all['Low']), 
                            np.maximum(abs(df_all['High'] - df_all['prev_close']), 
                            abs(df_all['prev_close'] - df_all['Low'])))
            
        _ticker = _ticker.replace('/','') 
        df = df_all[df_all.Symbol == _ticker].copy()
        # print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,_ticker,col.Style.RESET_ALL,end='',flush=True)
        df.loc[df.Symbol==_ticker,'ATR_'+str(_window)] = Wilder(df['TR'], _window)
        return(df.sort_index(axis=0))       

    def pivot(weekly_all, TICKER_LIST):
        # print(col.Fore.MAGENTA+'\nCalcul des PIVOT, RESISTANCE ET SUPPORT'+col.Style.RESET_ALL)
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        weekly_temp = weekly_all.copy()
        weekly_temp['PP'] = (weekly_temp.High.shift(1) + weekly_temp.Low.shift(1) + weekly_temp.Close.shift(1)) / 3
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
        return(weekly_temp.sort_index(axis=0))

    def pivotimportdf(df_all,weekly_all,TICKER_LIST):
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        weekly = weekly_all.copy()
        hourly = df_all.copy()
        weekly['Date'] = pd.to_datetime(weekly.Date)
        hourly['Date'] = pd.to_datetime(hourly.Date)
        hourly = hourly.join(weekly[['PP','S38','S62','S100','S138','S162','S200','R38','R62','R100','R138','R162','R200','Date']],how='left',on='Date',rsuffix='_2drop')
        hourly = hourly.drop(['Date_2drop'],axis=1)
        hourly.PP.fillna(method='ffill', inplace=True)
        hourly.S38.fillna(method='ffill', inplace=True)
        hourly.S62.fillna(method='ffill', inplace=True)
        hourly.S100.fillna(method='ffill', inplace=True)
        hourly.S138.fillna(method='ffill', inplace=True)
        hourly.S162.fillna(method='ffill', inplace=True)
        hourly.S200.fillna(method='ffill', inplace=True)
        hourly.R38.fillna(method='ffill', inplace=True)
        hourly.R62.fillna(method='ffill', inplace=True)
        hourly.R100.fillna(method='ffill', inplace=True)
        hourly.R138.fillna(method='ffill', inplace=True)
        hourly.R162.fillna(method='ffill', inplace=True)
        hourly.R200.fillna(method='ffill', inplace=True)
        return(hourly.sort_index(axis=0))

    def adr(daily_all,_window):
        # print(col.Fore.MAGENTA+'\nCalcul du ADR'+col.Style.RESET_ALL)
        
        daily = daily_all.copy()
        daily['ADR'] = (daily.High - daily.Low).rolling(_window).mean().shift(1)
        daily = daily.drop(['list','Week','WeekNo','Year'],axis=1)
        return(daily.sort_index(axis=0))

    def rvi(df_all,TICKER_LIST,_window):
        # print(col.Fore.MAGENTA+'\nCalcul du RVI'+col.Style.RESET_ALL)
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')

        df_all['Std'] = df_all.Close.rolling(window=_window).std()
        df_all['Positive'] = np.where((df_all.Std > df_all.Std.shift(1)),df_all.Std,0)
        df_all['Negative'] = np.where((df_all.Std < df_all.Std.shift(1)),df_all.Std,0)
        df_all['PoMA'] = Wilder(df_all['Positive'],_window)
        df_all['NeMA'] = Wilder(df_all['Negative'],_window)
        df_all['RVI'] = (100 * df_all['PoMA']) / (df_all['PoMA'] + df_all['NeMA'])
        df_all = df_all.drop(['Std','Positive','Negative','PoMA','NeMA'],axis=1)
        return(df_all.sort_index(axis=0))

    def getadr(daily_all,df_all, TICKER_LIST):
        # print("\nRécupération de l'ADR en cours...")
        _ticker = TICKER_LIST[0]
        _suffix='_2Drop'
        _ticker = _ticker.replace('/','')
        other = daily_all.copy()
        hourly = df_all.copy()
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
        #return(hourly.sort_index(axis=0))
        hourly['ADR'].fillna(method='ffill', inplace=True)
        return(hourly.sort_index(axis=0))

    def adrhnl(daily_all,df_all,TICKER_LIST):
        # print(col.Fore.CYAN+'\nCalcul du ADR High & Low'+col.Style.RESET_ALL)
        # print('En cours...')
        global _flagh, _flagl , val
        _ticker = TICKER_LIST[0]
        _flagh = 0
        _flagl = 0
        val = 0
        _ticker = _ticker.replace('/','')
        daily = daily_all.copy()
        hourly = df_all.copy()

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

        hourly = hourly.drop(['DateShiftMinus','DateShiftPlus','HighShift','LowShift','HighSlope','LowSlope'],axis=1)
        return(hourly.sort_index(axis=0))

    def sbgamma(df_all,TICKER_LIST):
        _ticker = TICKER_LIST[0]
        _ticker = _ticker.replace('/','')
        hourly = df_all[df_all.Symbol==_ticker].copy()
        hourly['SB_Gamma'] = (hourly.Close - hourly.Open)/(hourly.Close.shift(1) - hourly.Open.shift(1)) 
        return(hourly.sort_index(axis=0))

    def importohlc(df_all,other_all,TICKER_LIST,_suffix):
        # print('Récupération des OHLC en cours...')
        _ticker = TICKER_LIST[0]
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
                
        hourly['Open'+_suffix].fillna(method='ffill', inplace=True)
        hourly['High'+_suffix].fillna(method='ffill', inplace=True)
        hourly['Low'+_suffix].fillna(method='ffill', inplace=True)
        hourly['Close'+_suffix].fillna(method='ffill', inplace=True)
        return(hourly.sort_index(axis=0))

    def featuring(df_all):
        """[Entrer la df préparée avec les bons indicteurs. Renvoie une nouvelle df qu'avec les features + Symbol + Date + Signal]

        Args:
            df_all ([dataframe]): [Mettre la df qui doit être featurée.]
        """   
        features = pd.DataFrame()
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
        features['FOPENWEEKLY'] = df_all['Close'] - df_all['Open_weekly']
        features['FHIGHWEEKLY'] = df_all['Close'] - df_all['High_weekly']
        features['FLOWWEEKLY'] = df_all['Close'] - df_all['Low_weekly']
        features['FCLOSEWEEKLY'] = df_all['Close'] - df_all['Close_weekly']
        features['FOPENDAILY'] = df_all['Close'] - df_all['Open_daily']
        features['FHIGHDAILY'] = df_all['Close'] - df_all['High_daily']
        features['FLOWDAILY'] = df_all['Close'] - df_all['Low_daily']
        features['FCLOSEDAILY'] = df_all['Close'] - df_all['Close_daily']
        features['FOPENHOURLY'] = df_all['Close'] - df_all['Open_daily']
        features['FHIGHHOURLY'] = df_all['Close'] - df_all['High_daily']
        features['FLOWHOURLY'] = df_all['Close'] - df_all['Low_daily']
        features['FCLOSEHOURLY'] = df_all['Close'] - df_all['Close_daily']
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

    def scaling(features,scaler):
        """[Entrer la df deja featuree pour effectuer dessus le scaling MinMax. Renvoie la df actuelle => possibilité d'écraser]

        Args:
            features ([DataFrame]): [La dataframe qui a déjà été featurée]
            scaler ([type]): [description]
        """    
        features = features.dropna()

        features['FEMA_21'] = scaler.fit_transform(np.nan_to_num(features.FEMA_21.astype(np.float32)).reshape(-1, 1))
        features['FEMA_8'] = scaler.fit_transform(np.nan_to_num(features.FEMA_8.astype(np.float32)).reshape(-1, 1))
        features['FADRLo'] = scaler.fit_transform(np.nan_to_num(features.FADRLo.astype(np.float32)).reshape(-1, 1))
        features['FADRHi'] = scaler.fit_transform(np.nan_to_num(features.FADRHi.astype(np.float32)).reshape(-1, 1))
        features['FRVI40'] = scaler.fit_transform(np.nan_to_num(features.FRVI40.astype(np.float32)).reshape(-1, 1))
        features['FRVI60'] = scaler.fit_transform(np.nan_to_num(features.FRVI60.astype(np.float32)).reshape(-1, 1))
        features['FONLOSMA5'] = scaler.fit_transform(np.nan_to_num(features.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
        features['FONHISMA5'] = scaler.fit_transform(np.nan_to_num(features.FONHISMA5.astype(np.float32)).reshape(-1, 1))
        features['FONLOSMA21'] = scaler.fit_transform(np.nan_to_num(features.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
        features['FONHISMA21'] = scaler.fit_transform(np.nan_to_num(features.FONHISMA21.astype(np.float32)).reshape(-1, 1))
        features['FONLOSMA34'] = scaler.fit_transform(np.nan_to_num(features.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
        features['FSBGAMMA'] = scaler.fit_transform(np.nan_to_num(features.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
        features['FOPENWEEKLY'] = scaler.fit_transform(np.nan_to_num(features.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FHIGHWEEKLY'] = scaler.fit_transform(np.nan_to_num(features.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FLOWWEEKLY'] = scaler.fit_transform(np.nan_to_num(features.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FCLOSEWEEKLY'] = scaler.fit_transform(np.nan_to_num(features.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FOPENDAILY'] = scaler.fit_transform(np.nan_to_num(features.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
        features['FHIGHDAILY'] = scaler.fit_transform(np.nan_to_num(features.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
        features['FLOWDAILY'] = scaler.fit_transform(np.nan_to_num(features.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
        features['FCLOSEDAILY'] = scaler.fit_transform(np.nan_to_num(features.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
        features['FOPENHOURLY'] = scaler.fit_transform(np.nan_to_num(features.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FHIGHHOURLY'] = scaler.fit_transform(np.nan_to_num(features.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FLOWHOURLY'] = scaler.fit_transform(np.nan_to_num(features.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FCLOSEHOURLY'] = scaler.fit_transform(np.nan_to_num(features.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FSMA200'] = scaler.fit_transform(np.nan_to_num(features.FSMA200.astype(np.float32)).reshape(-1, 1))
        features['FBOLUP20'] = scaler.fit_transform(np.nan_to_num(features.FBOLUP20.astype(np.float32)).reshape(-1, 1))
        features['FPP'] = scaler.fit_transform(np.nan_to_num(features.FPP.astype(np.float32)).reshape(-1, 1))
        features['FS38'] = scaler.fit_transform(np.nan_to_num(features.FS38.astype(np.float32)).reshape(-1, 1))
        features['FS62'] = scaler.fit_transform(np.nan_to_num(features.FS62.astype(np.float32)).reshape(-1, 1))
        features['FS100'] = scaler.fit_transform(np.nan_to_num(features.FS100.astype(np.float32)).reshape(-1, 1))
        features['FS138'] = scaler.fit_transform(np.nan_to_num(features.FS138.astype(np.float32)).reshape(-1, 1))
        features['FR162'] = scaler.fit_transform(np.nan_to_num(features.FS162.astype(np.float32)).reshape(-1, 1))
        features['FS200'] = scaler.fit_transform(np.nan_to_num(features.FS200.astype(np.float32)).reshape(-1, 1))
        features['FR38'] = scaler.fit_transform(np.nan_to_num(features.FR38.astype(np.float32)).reshape(-1, 1))
        features['FR62'] = scaler.fit_transform(np.nan_to_num(features.FR62.astype(np.float32)).reshape(-1, 1))
        features['FR100'] = scaler.fit_transform(np.nan_to_num(features.FR100.astype(np.float32)).reshape(-1, 1))
        features['FR138'] = scaler.fit_transform(np.nan_to_num(features.FR138.astype(np.float32)).reshape(-1, 1))
        features['FR162'] = scaler.fit_transform(np.nan_to_num(features.FR162.astype(np.float32)).reshape(-1, 1))
        features['FR200'] = scaler.fit_transform(np.nan_to_num(features.FR200.astype(np.float32)).reshape(-1, 1))
        features['SBATR'] = scaler.fit_transform(np.nan_to_num(features.SBATR.astype(np.float32)).reshape(-1, 1))
        
        
        return(features)

    def quantile(features,quantile_transform):
        """[Transformation par les Quantile]

        Args:
            features ([dataframe]): [dataframe]
            quantile_transform ([sklearn]): [from preprocessing]
        """    
        
        features['FEMA_21'] = quantile_transform(np.nan_to_num(features.FEMA_21.astype(np.float32)).reshape(-1, 1))
        features['FEMA_8'] = quantile_transform(np.nan_to_num(features.FEMA_8.astype(np.float32)).reshape(-1, 1))
        features['FADRLo'] = quantile_transform(np.nan_to_num(features.FADRLo.astype(np.float32)).reshape(-1, 1))
        features['FADRHi'] = quantile_transform(np.nan_to_num(features.FADRHi.astype(np.float32)).reshape(-1, 1))
        features['FRVI40'] = quantile_transform(np.nan_to_num(features.FRVI40.astype(np.float32)).reshape(-1, 1))
        features['FRVI60'] = quantile_transform(np.nan_to_num(features.FRVI60.astype(np.float32)).reshape(-1, 1))
        features['FONLOSMA5'] = quantile_transform(np.nan_to_num(features.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
        features['FONHISMA5'] = quantile_transform(np.nan_to_num(features.FONHISMA5.astype(np.float32)).reshape(-1, 1))
        features['FONLOSMA21'] = quantile_transform(np.nan_to_num(features.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
        features['FONHISMA21'] = quantile_transform(np.nan_to_num(features.FONHISMA21.astype(np.float32)).reshape(-1, 1))
        features['FONLOSMA34'] = quantile_transform(np.nan_to_num(features.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
        features['FSBGAMMA'] = quantile_transform(np.nan_to_num(features.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
        features['FOPENWEEKLY'] = quantile_transform(np.nan_to_num(features.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FHIGHWEEKLY'] = quantile_transform(np.nan_to_num(features.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FLOWWEEKLY'] = quantile_transform(np.nan_to_num(features.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FCLOSEWEEKLY'] = quantile_transform(np.nan_to_num(features.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
        features['FOPENDAILY'] = quantile_transform(np.nan_to_num(features.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
        features['FHIGHDAILY'] = quantile_transform(np.nan_to_num(features.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
        features['FLOWDAILY'] = quantile_transform(np.nan_to_num(features.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
        features['FCLOSEDAILY'] = quantile_transform(np.nan_to_num(features.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
        features['FOPENHOURLY'] = quantile_transform(np.nan_to_num(features.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FHIGHHOURLY'] = quantile_transform(np.nan_to_num(features.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FLOWHOURLY'] = quantile_transform(np.nan_to_num(features.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FCLOSEHOURLY'] = quantile_transform(np.nan_to_num(features.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
        features['FSMA200'] = quantile_transform(np.nan_to_num(features.FSMA200.astype(np.float32)).reshape(-1, 1))
        features['FBOLUP20'] = quantile_transform(np.nan_to_num(features.FBOLUP20.astype(np.float32)).reshape(-1, 1))
        features['FPP'] = quantile_transform(np.nan_to_num(features.FPP.astype(np.float32)).reshape(-1, 1))
        features['FS38'] = quantile_transform(np.nan_to_num(features.FS38.astype(np.float32)).reshape(-1, 1))
        features['FS62'] = quantile_transform(np.nan_to_num(features.FS62.astype(np.float32)).reshape(-1, 1))
        features['FS100'] = quantile_transform(np.nan_to_num(features.FS100.astype(np.float32)).reshape(-1, 1))
        features['FS138'] = quantile_transform(np.nan_to_num(features.FS138.astype(np.float32)).reshape(-1, 1))
        features['FR162'] = quantile_transform(np.nan_to_num(features.FS162.astype(np.float32)).reshape(-1, 1))
        features['FS200'] = quantile_transform(np.nan_to_num(features.FS200.astype(np.float32)).reshape(-1, 1))
        features['FR38'] = quantile_transform(np.nan_to_num(features.FR38.astype(np.float32)).reshape(-1, 1))
        features['FR62'] = quantile_transform(np.nan_to_num(features.FR62.astype(np.float32)).reshape(-1, 1))
        features['FR100'] = quantile_transform(np.nan_to_num(features.FR100.astype(np.float32)).reshape(-1, 1))
        features['FR138'] = quantile_transform(np.nan_to_num(features.FR138.astype(np.float32)).reshape(-1, 1))
        features['FR162'] = quantile_transform(np.nan_to_num(features.FR162.astype(np.float32)).reshape(-1, 1))
        features['FR200'] = quantile_transform(np.nan_to_num(features.FR200.astype(np.float32)).reshape(-1, 1))
        features['SBATR'] = quantile_transform(np.nan_to_num(features.SBATR.astype(np.float32)).reshape(-1, 1))
        return(features)

    def stochastic(df):
        """[Determine the stochastic strategy based on _window=5. Enter dataframe with column slow_K5 and slow_D5. Return dataframe with column Signal.]

        Args:
            df ([dataframe]): [Must been already computed with slow_D5 and slow_K5 column]
        """    
        ##### CONDITIONS LONG
        _condition_1 = (df.slow_K5 < 20) & (df.slow_K5.shift(1) < df.slow_D5.shift(1)) & (df.slow_K5 > df.slow_D5)

        ##### CONDITIONS SHORT
        _condition_1_bar = (df.slow_K5 > 80) & (df.slow_K5.shift(1) > df.slow_D5.shift(1)) & (df.slow_K5 < df.slow_D5)

        ##### 1 condition
        df['Signal'] = np.where(_condition_1,1,np.where(_condition_1_bar,-1,0))
        return(df) 
    
    def conX(con):
        try:
            con.is_connected()
            if con.is_connected() == True:
                print('Déjà connecté')
                engine.say("already Connected")
                engine.runAndWait()
                print('')
            else:
                con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
                print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
                print('Compte utilisé : ',con.get_account_ids())
                engine.say("Bigfoot is Connected")
                engine.runAndWait()
                print('')
        except:
            con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
            if con.is_connected() == True:
                print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
                print('Compte utilisé : ',con.get_account_ids())
                engine.say("Bigfoot is Connected")
                engine.runAndWait()
                print('')
            else:
                print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
                engine.say("Mayday, mayday, Not Connected, mauzerfucker!")
                engine.say("Check your internet, and launch agin the Bigfoot")
                engine.runAndWait()
                print('')
        return(con)

    def buy(_period):
        print(dt.datetime.now())
        ##### BUY 
        _price = round(con.get_candles(x,period=_period,number=1).askclose[-1],5)
        _time = con.get_candles(x,period=_period,number=1).index[-1]
        _amount = _cash_ini / df_ratefx.loc[x[:3],'rate']
        _limit = round(_price + _price * 0.002,5)
        _stop = round(_price  - _price * 0.001,5)
        order = con.open_trade(symbol=x,is_buy=True, is_in_pips=False, amount=_amount, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)
        print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
        print(col.Fore.GREEN,'Achat sur le ticker',col.Fore.YELLOW,x,col.Fore.GREEN,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

    def sell(_period):
        print(dt.datetime.now())
        _price = round(con.get_candles(x,period=_period,number=1).bidclose[-1],5)
        _time = con.get_candles(x,period=_period,number=1).index[-1]
        _amount = _cash_ini / df_ratefx.loc[x[:3],'rate']
        _stop = round(_price + _price * 0.001,5)
        _limit = round(_price  - _price * 0.002,5)
        order = con.open_trade(symbol=x,is_buy=False, is_in_pips=False, amount=_amount, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)
        print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
        print(col.Fore.RED,'Vente sur le ticker',col.Fore.YELLOW,x,col.Fore.RED,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

        #expiration = (dt.datetime.now() + dt.timedelta(hours=4)).strftime(format='%Y-%m-%d %H:%M'))
        return()

        ___Author___='LumberJack Jyss'
    

    ############################################################################
    ############################################################################
    #################     M A I N      F U N C T I O N      ####################
    ############################################################################
    ############################################################################


    print('Global Optimized LumberJack Environment Motor for For_Ex\nLumberJack Jyss 5781(c)')
    print(col.Fore.BLUE,'°0Oo_D.A.G._26_oO0°')
    print(col.Fore.YELLOW,col.Back.BLUE,'--- Bigfoot 1. #v0.70 ---',col.Style.RESET_ALL)


    print('')
    engine.say(" Initialization of Bigfoot 1, FX system")
    engine.say("Bigfoot's Connexion to the a p i")
    engine.runAndWait()

    _pnl_max = 0
    _pnl_min = 0

    # Connexion to the api
    try:
        con
    except NameError:
        con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
        engine.say("Bigfoot is Connected")
        engine.runAndWait()
                
    else:
        con = conX(con)

    print('\rChargement de la base...',end='',flush=True)
    engine.say("Ignition of Bigfoot. Loading the database.")
    engine.runAndWait()

    # Loading the saved model of MLP Classifier
    savename = 'Save_'+_ticker+'_'+_period+'.sav'
    _model = joblib.load(savename)
    # Loading all the stuff got from HDD added with previous version
    df_all = joblib.load(_ticker+'_'+_period)
    df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
    hourly_all = joblib.load(_ticker+'_H1')
    hourly_all = hourly_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]

    print('\r.....................Base Chargée.',end='',flush=True)

    engine.say("Databases are loaded. Ready to enter Live")
    engine.runAndWait()

    # Entering the endless loop    
    while True:

        engine.say("Building the base")
        engine.runAndWait()
        print('\nConstruction de la base...')
        ##########
        
        print(x)
        if len(con.get_open_positions()) == 0:
            if _poz==1:
                print(col.Fore.MAGENTA,'Position pour',x,' ouverte à',con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].open.values[-1],\
                    'et cloturée à',con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].close.values[-1],col.Style.RESET_ALL)
            _poz = 0
            print('Pas de position ouverte pour le moment')

        elif x in set(con.get_open_positions().currency.to_list()):
            _poz = 1
            print(col.Back.LIGHTRED_EX,col.Fore.LIGHTCYAN_EX,'/!\ ATTENTION',x,' EST DEJA EN POSITION /!\ ',col.Style.RESET_ALL)
            _tradeID = con.get_open_positions()[con.get_open_positions().currency==x].tradeId.values[-1]
            print('TradeID =',_tradeID)
            try:
                if con.get_summary()[con.get_summary().currency==x].avgBuy.values[-1] > 0:
                    _sens = ' Long'
                elif con.get_summary()[con.get_summary().currency==x].avgSell.values[-1] < 0:
                    _sens = ' Short'
                _pnl = con.get_summary()[con.get_summary().currency==x].grossPL.values[-1]
                if _pnl > _pnl_max : 
                    _pnl_max = _pnl
                if _pnl_min == 0:
                    _pnl_min = _pnl
                if _pnl_max == 0:
                    _pnl_max = _pnl
                if _pnl < _pnl_min:
                    _pnl_min = _pnl
                if  _pnl>=0:
                    print(col.Fore.GREEN,'P&L (',_sens,') en cours :',_pnl,col.Style.RESET_ALL)
                    print(' (Max :',_pnl_max,')')
                    print(' (Min :',_pnl_min,')')
                else:
                    print(col.Fore.RED,'P&L (',_sens,') en cours :',_pnl,col.Style.RESET_ALL)
                    print(' (Max :',_pnl_max,')')
                    print(' (Min :',_pnl_min,')')
            except:
                print('Plus de position détectée pour',x)
                print('Etat des positions :',con.get_open_positions_summary())
        else:
            if _poz==1:
                print(col.Fore.MAGENTA,'Position pour',x,' ouverte à',con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].open.values[-1],\
                    'et cloturée à',con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].close.values[-1],col.Style.RESET_ALL)
            _poz = 0
            print('Pas de position ouverte pour',x)
        
        print('\nWaiting for the candle...')
        print()

        ##########
        # Wait until the minute is in the 'm5' list
        while dt.datetime.now().minute not in [0,5,10,15,20,25,30,35,40,45,50,55]:
            print('\rTicker tracké :',x,' ',dt.datetime.now(),end='',flush=True)
            time.sleep(1)
        print('\nGo!')
        time.sleep(10)
        print()
        # Set the end & the beguinning of scraping, from last index of df_all to now    
        _fin = dt.datetime.now()
        _deb = df_all.index[-1]
        _debut = dt.datetime(_deb.year,_deb.month,_deb.day,_deb.hour,_deb.minute)
        # Scrap the addon & build it to be compliant with our df_all
        addon = con.get_candles(x,period=_period,start=_debut,end=_fin).drop(['tickqty'],axis=1)
        addon = addon.rename(columns={'bidopen':'OpenBid','bidclose':'CloseBid','bidhigh':'HighBid','bidlow':'LowBid','askopen':'OpenAsk','askclose':'CloseAsk','askhigh':'HighAsk','asklow':'LowAsk'})
        # Calculate the mid prices
        addon['Open'] = (addon.OpenAsk + addon.OpenBid)/2
        addon['High'] = (addon.HighAsk + addon.HighBid)/2
        addon['Low'] = (addon.LowAsk + addon.LowBid)/2
        addon['Close'] = (addon.CloseAsk + addon.CloseBid)/2
        addon['Symbol'] = x.replace('/','')
        addon['Date'] = addon.index
        addon['Date'] = pd.to_datetime(addon['Date'].dt.strftime(date_format='%Y-%m-%d'))
        df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
        # Concatenate the bases
        df_all = df_all.append(addon.iloc[1:,:])
        #df_all = df_all.iloc[-263570:,:]
        
        # Make a 'day' column
        df_all = timerange1D(df_all)

        # Do the same thing with hourly that we have done with df_all
        _fin = dt.datetime.now()
        _deb = hourly_all.index[-1]
        _debut = dt.datetime(_deb.year,_deb.month,_deb.day,_deb.hour)
        hourly_add = con.get_candles(x,period='H1',start=_debut,end=_fin).drop(['tickqty'],axis=1) # df_all[df_all.index.minute==0] # scrap_hist(x)
        hourly_add = hourly_add.rename(columns={'bidopen':'OpenBid','bidclose':'CloseBid','bidhigh':'HighBid','bidlow':'LowBid','askopen':'OpenAsk','askclose':'CloseAsk','askhigh':'HighAsk','asklow':'LowAsk'})
        hourly_add['Open'] = (hourly_add.OpenAsk + hourly_add.OpenBid)/2
        hourly_add['High'] = (hourly_add.HighAsk + hourly_add.HighBid)/2
        hourly_add['Low'] = (hourly_add.LowAsk + hourly_add.LowBid)/2
        hourly_add['Close'] = (hourly_add.CloseAsk + hourly_add.CloseBid)/2
        hourly_add['Symbol'] = x.replace('/','')
        hourly_all = hourly_all.append(hourly_add.iloc[1:,:])
        hourly_all = timerange1D(hourly_all)

        # Make daily & weekly bases
        daily_all = get_daily(hourly_all,TICKER_LIST)
        daily_all = timerange1W(daily_all)
        weekly_all = get_weekly(daily_all,TICKER_LIST)

        # Calculate the indicators
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
        
        # Apply the raw strategy to get the signals
        df_all = stochastic(df_all)
        
        # Featuring
        features = featuring(df_all)

        # And drop the nan
        features = features.dropna()

        # Proceed an MaxAbsScaler on features
        features = scaling(features,scaler=_scaler)

        # Apply quantilation
        features = quantile(features,quantile_transform)

        # Apply our Deep Learning model to generate real signals
        _valid = _model.predict(features.drop(['Date','Symbol','Signal'],axis=1))[-1]

        _signal = df_all.Signal[-1]

        if df_all.index[-1] != features.index[-1]:
            print(col.Back.LIGHTRED_EX,col.Fore.LIGHTCYAN_EX,'/!\ ATTENTION DECALAGE DES DERNIERES VALEURS /!\ ',col.Style.RESET_ALL)
            print('Dernier index de df_all',df_all.index[-1])
            print('Dernier index de features',features.index[-1])
            print('Dernière bougie :')
            print(df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']].iloc[-1:,:])

        else:

            print('\nTest sur la bougie',features.index[-1])
            if _valid == 1 and _signal == 1 and _poz == 0 :
                buy(_period)

            elif _valid == 1 and _signal == -1 and _poz == 0 :
                sell(_period)

            else:
                print(col.Fore.BLUE,'\nNO SIGNAL FOR',col.Fore.YELLOW,x,'\n',col.Style.RESET_ALL)
                print('Raw signal to',_signal)
                print(df_all[['slow_K5','slow_D5']].tail(2))
        
        print('Reset of df_all')

        # Save the base
        joblib.dump(df_all , _ticker+'_'+_period)
        joblib.dump(hourly_all , _ticker+'_H1')

        # Purification
        df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
        hourly_all = hourly_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]  

if __name__ == "__main__":
    pass
