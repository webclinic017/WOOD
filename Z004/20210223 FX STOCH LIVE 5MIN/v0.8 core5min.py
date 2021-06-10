__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
####### RECUPERATION DONNEES ET PREPARATION DES DATA FX ############
####################################################################
####################################################################
        
import time
from colorama.ansi import Style
from colorama.initialise import reset_all     
import pandas as pd
import numpy as np
import colorama as col
import joblib
import datetime as dt
import pyttsx3
import datetime as dt
#from sklearn.neural_network import MLPClassifier
engine = pyttsx3.init()

from slack_sdk import WebClient
_slack_token = joblib.load('_slack_token')
client = WebClient(token=_slack_token)


def tweet_it():
    import twitter
    _ticker = 'à définir'
    api = twitter.Api(consumer_key='KXALdSMKE1dN4lNCYNxhWgfyU',
                        consumer_secret='xGZjEoBMKjoxxAfjsmRa4iUvxAsI3co5IyjUOpNxkhL08KZKJk',
                        access_token_key='1029626801939726337-jrbQ0vUlOGc9wI4TlmpQItPwBKrhMX',
                        access_token_secret='ea8mKXpINq2dtAmhZNAXPXQMTGLlEATmmgatQJfAOATPV')
    user = "@Go!em"
    statuses = api.GetUserTimeline(screen_name=('@Golem_FX'))
    # ecrire message
    api.PostUpdate('INVERSE CLOSE '+_ticker+' horodaté à '+str(dt.datetime.now()))


# _user_id = 'D261290212' # 'D261282181'
# _compte = '01285488' # '01215060'
# _password = '0Iqqy' # 'waXz1'
# _token = '7ac07a01bd599b6063a75be72a4b56edb22afa7c'

def golive(_period='m5',TICKER_LIST=['EUR/USD'],_target=0.002,_sl=0.001,_cash_ini=10,_token='17df343a1048b2b61158f1b1de606a14413a2113',_server='demo',_trigger_spread=0.035):
    global fxcmpy
    import fxcmpy
    print('version fxcmpy :',fxcmpy.__version__)
    
    def get_daily(df_all):
        daily_all = pd.DataFrame(index=df_all.Date.unique())
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
        return(daily_all.sort_index(axis=0))

    def get_weekly(daily_all):        
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
        return(weekly_all.sort_index(axis=0))

    def timerange1D(df_all):
        df_all['Date'] = df_all.index
        df_all['Date'] = df_all['Date'].dt.strftime(date_format='%Y-%m-%d')
        return(df_all.sort_index(axis=0))

    def timerange1W(daily_all):
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
        start = np.where(~np.isnan(data))[0][0] # Positionne après les nan
        Wilder = np.array([np.nan]*len(data)) # Replace les nan en début de liste pour ne pas changer la longueur
        Wilder[start+window-1] = data[start:(start+window)].mean() #Simple Moving Average pour la window window
        for i in range(start+window,len(data)):
            Wilder[i] = ((Wilder[i-1]*(window-1) + data[i])/window) #Wilder Smoothing
        return(Wilder)

    def ema(df_all, _window):
        df_all['EMA_'+str(_window)] = df_all.Close.ewm(span=_window,adjust=False).mean()
        return(df_all.sort_index(axis=0))

    def sma(df_all,_window=200):
        df_all['SMA_'+str(_window)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window = _window).mean())
        return(df_all.sort_index(axis=0))

    def slowstochastic(df_all,_window=5,_per=3):
        df_all['Lowest_'+str(_window)] = df_all['Low'].transform(lambda x: x.rolling(window = _window).min())
        df_all['Highest_'+str(_window)] = df_all['High'].transform(lambda x: x.rolling(window = _window).max())
        df_all['slow_K'+str(_window)] = (((df_all['Close'] - df_all['Lowest_'+str(_window)])/(df_all['Highest_'+str(_window)] - df_all['Lowest_'+str(_window)]))*100).rolling(window = _per).mean()
        df_all['slow_D'+str(_window)] = df_all['slow_K'+str(_window)].rolling(window = _per).mean()
        df_all = df_all.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)
        return(df_all.sort_index(axis=0))

    def bollinger(df_all,_slow=15):
        df_all['MA'+str(_slow)] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=_slow).mean())
        df_all['SD'] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=_slow).std())
        df_all['UpperBand'] = df_all['MA'+str(_slow)] + 2*df_all['SD']
        df_all['LowerBand'] = df_all['MA'+str(_slow)] - 2*df_all['SD']
        df_all = df_all.drop(['MA'+str(_slow),'SD'],axis=1)
        return(df_all.sort_index(axis=0))

    def onlosma(df_all,_window=8):
        df_all['ONLOSMA_'+str(_window)] = df_all.Low.rolling(_window).mean()
        return(df_all.sort_index(axis=0))
        
    def onhisma(df_all,_window=8):
        df_all['ONHISMA_'+str(_window)] = df_all.High.rolling(_window).mean()
        return(df_all.sort_index(axis=0))

    def atr(df_all,_window=14):
        df_all['prev_close'] = df_all['Close'].shift(1)

        df_all['TR'] = np.maximum((df_all['High'] - df_all['Low']), 
                            np.maximum(abs(df_all['High'] - df_all['prev_close']), 
                            abs(df_all['prev_close'] - df_all['Low'])))

        df_all.loc[df_all.Symbol==_ticker,'ATR_'+str(_window)] = Wilder(df_all['TR'], _window)
        return(df_all.sort_index(axis=0))       

    def pivot(weekly_all):
        weekly_all['PP'] = (weekly_all.High.shift(1) + weekly_all.Low.shift(1) + weekly_all.Close.shift(1)) / 3
        weekly_all['S38'] = weekly_all.PP - (0.382 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['S62'] = weekly_all.PP - (0.618 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['S100'] = weekly_all.PP - (1 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['S138'] = weekly_all.PP - (1.382 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['S162'] = weekly_all.PP - (1.618 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['S200'] = weekly_all.PP - (2 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['R38'] = weekly_all.PP + (0.382 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['R62'] = weekly_all.PP + (0.618 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['R100'] = weekly_all.PP + (1 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['R138'] = weekly_all.PP + (1.382 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['R162'] = weekly_all.PP + (1.618 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        weekly_all['R200'] = weekly_all.PP + (2 * (weekly_all.High.shift(1) - weekly_all.Low.shift(1)))
        return(weekly_all.sort_index(axis=0))

    def pivotimportdf(df_all,weekly_all):
        weekly_all['Date'] = pd.to_datetime(weekly_all.Date)
        df_all['Date'] = pd.to_datetime(df_all.Date)
        df_all = df_all.join(weekly_all[['PP','S38','S62','S100','S138','S162','S200','R38','R62','R100','R138','R162','R200','Date']],how='left',on='Date',rsuffix='_2drop')
        df_all = df_all.drop(['Date_2drop'],axis=1)
        df_all.PP.fillna(method='ffill', inplace=True)
        df_all.S38.fillna(method='ffill', inplace=True)
        df_all.S62.fillna(method='ffill', inplace=True)
        df_all.S100.fillna(method='ffill', inplace=True)
        df_all.S138.fillna(method='ffill', inplace=True)
        df_all.S162.fillna(method='ffill', inplace=True)
        df_all.S200.fillna(method='ffill', inplace=True)
        df_all.R38.fillna(method='ffill', inplace=True)
        df_all.R62.fillna(method='ffill', inplace=True)
        df_all.R100.fillna(method='ffill', inplace=True)
        df_all.R138.fillna(method='ffill', inplace=True)
        df_all.R162.fillna(method='ffill', inplace=True)
        df_all.R200.fillna(method='ffill', inplace=True)
        return(df_all.sort_index(axis=0))

    def adr(daily_all,_window):
        daily_all['ADR'] = (daily_all.High - daily_all.Low).rolling(_window).mean().shift(1)
        daily_all = daily_all.drop(['list','Week','WeekNo','Year'],axis=1)
        return(daily_all.sort_index(axis=0))

    def rvi(df_all,_window):
        df_all['Std'] = df_all.Close.rolling(window=_window).std()
        df_all['Positive'] = np.where((df_all.Std > df_all.Std.shift(1)),df_all.Std,0)
        df_all['Negative'] = np.where((df_all.Std < df_all.Std.shift(1)),df_all.Std,0)
        df_all['PoMA'] = Wilder(df_all['Positive'],_window)
        df_all['NeMA'] = Wilder(df_all['Negative'],_window)
        df_all['RVI'] = (100 * df_all['PoMA']) / (df_all['PoMA'] + df_all['NeMA'])
        df_all = df_all.drop(['Std','Positive','Negative','PoMA','NeMA'],axis=1)
        return(df_all.sort_index(axis=0))

    def getadr(daily_all,df_all):
        _suffix='_2Drop'
        daily_all['Date'] = pd.to_datetime(daily_all.Date)
        df_all['Date'] = pd.to_datetime(df_all.Date)
        df_all = df_all.join(daily_all[['ADR']],how='left',on='Date',rsuffix=_suffix)
        df_all = df_all.join(daily_all[['High']],how='left',on='Date',rsuffix=_suffix)
        df_all = df_all.join(daily_all[['Low']],how='left',on='Date',rsuffix=_suffix)
        df_all = df_all.rename(columns={'High'+_suffix: "DayHigh", 'Low'+_suffix: "DayLow"})
        try:
            df_all = df_all.drop(['Date'+_suffix],axis=1)
        except:
            pass
        df_all['ADR'].fillna(method='ffill', inplace=True)
        return(df_all.sort_index(axis=0))

    def adrhnl(df_all):
        global _flagh, _flagl , val
        _flagh = 0
        _flagl = 0
        val = 0

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

        
        df_all['DateShiftMinus'] = df_all.Date.shift(1)
        df_all['DateShiftPlus'] = df_all.Date.shift(-1)

        df_all['HighShift'] = df_all.High.shift(1)
        df_all['LowShift'] = df_all.Low.shift(1)

        df_all['HighSlope'] = df_all.apply(fh,axis=1)
        df_all['LowSlope'] = df_all.apply(fl,axis=1)
        df_all['HighSlope'].fillna(method='ffill', inplace=True)
        df_all['LowSlope'].fillna(method='ffill', inplace=True)

        df_all['ADR_High'] = df_all.LowSlope + df_all.ADR
        df_all['ADR_Low'] = df_all.HighSlope - df_all.ADR

        df_all = df_all.drop(['DateShiftMinus','DateShiftPlus','HighShift','LowShift','HighSlope','LowSlope'],axis=1)
        return(df_all.sort_index(axis=0))

    def sbgamma(df_all):      
        _op1 = (df_all.Close - df_all.Open)/(df_all.Close.shift(1) - df_all.Open.shift(1))
        _op2 = (df_all.Close - df_all.Open)/(df_all.CloseAsk.shift(1) - df_all.OpenAsk.shift(1))
        _op3 = (df_all.Close - df_all.Open)/(df_all.CloseBid.shift(1) - df_all.OpenBid.shift(1))
        _op4 = (df_all.Close - df_all.Open)/(df_all.CloseBid.shift(1) - df_all.OpenAsk.shift(1))
        _op5 = (df_all.Close - df_all.Open)/(df_all.CloseAsk.shift(1) - df_all.OpenBid.shift(1))

        _condition1 = df_all.Close.shift(1) != df_all.Open.shift(1)
        _condition2 = df_all.CloseAsk.shift(1) != df_all.OpenAsk.shift(1)
        _condition3 = df_all.CloseBid.shift(1) != df_all.OpenBid.shift(1)
        _condition4 = df_all.CloseBid.shift(1) != df_all.OpenAsk.shift(1)
        _condition5 = df_all.CloseAsk.shift(1) != df_all.OpenBid.shift(1)

        df_all['SB_Gamma'] = np.where(_condition1,_op1,np.where(_condition2,_op2,np.where(_condition3,_op3,np.where(_condition4,_op4,np.where(_condition5,_op5,1.93E13)))))
        return(df_all.sort_index(axis=0))

    def importohlc(df_all,other_all,_suffix):

        other_all['Date'] = pd.to_datetime(other_all.Date)
        df_all['Date'] = pd.to_datetime(df_all.Date)
        df_all = df_all.join(other_all[['Open','High','Low','Close']],how='left',on='Date',rsuffix=_suffix)
        try:
            df_all = df_all.drop(['Date'+_suffix],axis=1)
        except:
            pass
                
        df_all['Open'+_suffix].fillna(method='ffill', inplace=True)
        df_all['High'+_suffix].fillna(method='ffill', inplace=True)
        df_all['Low'+_suffix].fillna(method='ffill', inplace=True)
        df_all['Close'+_suffix].fillna(method='ffill', inplace=True)
        return(df_all.sort_index(axis=0))

    def featuring(df_all):  
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

    def scaling(features,scaler):
          
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
                engine.say("Gaulem is Connected")
                engine.runAndWait()
                print('')
        except:
            con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
            if con.is_connected() == True:
                print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
                print('Compte utilisé : ',con.get_account_ids())
                engine.say("Gaulem is Connected")
                engine.runAndWait()
                print('')
            else:
                print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
                engine.say("Mayday, mayday, Not Connected, mauzerfucker!")
                engine.say("Check your internet, and launch agin the Gaulem")
                engine.runAndWait()
                print('')
        return(con)

    def buy(_period):
        print(dt.datetime.now())
        ##### BUY 
        _price = round(con.get_candles(x,period=_period,number=1).askclose[-1],5)
        _time = con.get_candles(x,period=_period,number=1).index[-1]
        _amount = _cash_ini / df_ratefx.loc[x[:3],'rate']
        _limit = round(_price + _price * _target,5) # Take Profit
        _stop = round(_price  - _price * _sl,5) # Stop Loss
        order = con.open_trade(symbol=x,is_buy=True, is_in_pips=False, amount=_amount, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)#,trailing_step=1)
        print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
        print(col.Fore.GREEN,'Achat sur le ticker',col.Fore.YELLOW,x,col.Fore.GREEN,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

    def sell(_period):
        print(dt.datetime.now())
        _price = round(con.get_candles(x,period=_period,number=1).bidclose[-1],5)
        _time = con.get_candles(x,period=_period,number=1).index[-1]
        _amount = _cash_ini / df_ratefx.loc[x[:3],'rate']
        _stop = round(_price + _price * _sl,5) # Stop Loss
        _limit = round(_price  - _price * _target,5) # Take Profit
        order = con.open_trade(symbol=x,is_buy=False, is_in_pips=False, amount=_amount, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)#,trailing_step=1)
        print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
        print(col.Fore.RED,'Vente sur le ticker',col.Fore.YELLOW,x,col.Fore.RED,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

        #expiration = (dt.datetime.now() + dt.timedelta(hours=4)).strftime(format='%Y-%m-%d %H:%M'))
        return()
    
    def DagMaxBase(features_train):
        features_max = pd.DataFrame()
        for label in features_train.drop(['Date','Symbol'],axis=1).columns:
            features_max.loc[0,label] = abs(features[label]).max()   # max([abs(features_train[label]).max(),abs(features_train[label]).max()])
        joblib.dump(features_max,x.replace('/','')+'_MAX.dagmax')
        return(features_max)

    
    ############################################################################
    ############################################################################
    #################     M A I N      F U N C T I O N      ####################
    ############################################################################
    ############################################################################

    while True:

        print('Global Optimized LumberJack Environment Motor for For_Ex\nLumberJack Jyss 5781(c)')
        print(col.Fore.CYAN,'°0Oo_D.A.G._26_oO0°')
        print(col.Fore.YELLOW,col.Back.BLUE,'--- Golem FX #v0.80 ---',col.Style.RESET_ALL)


        print('')
        engine.say(" Initialization of Gaulem 1, FX system")
        engine.say("Gaulem's Connexion to the a p i")
        engine.runAndWait()
        
        x = TICKER_LIST[0]
        _ticker = x.replace('/','')

        _poz = 0
        try:
            _poz = joblib.load(x.replace('/','')+'_POZ.dag')
        except:
            pass

        _pnl_max = 0
        _pnl_min = 0
        if _poz != 0 :
            try:
                _pnl_max = joblib.load(x.replace('/','')+'_PNLMAX.dag')
                _pnl_min = joblib.load(x.replace('/','')+'_PNLMIN.dag')
            except:
                pass

        TIK = ['AUD','NZD','GBP','JPY','CHF','CAD','SEK','NOK','ILS','MXN','USD','EUR']
        RATE = [0.776,0.721,1.3912,1/105.91,1/0.892,1/1.2681,1/8.2884,1/8.4261,1/3.2385,1/20.1564,1,1/1.21]
        df_ratefx = pd.DataFrame(index=TIK)
        df_ratefx['rate'] = RATE

        features_max = joblib.load(x.replace('/','')+'_MAX.dagmax')

        # Connexion to the api
        try:
            con
        except NameError:
            con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
            print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
            print('Compte utilisé : ',con.get_account_ids())
            engine.say("Gaulem is Connected")
            engine.runAndWait()
                    
        else:
            con = conX(con)

        print('\rChargement de la base...',end='',flush=True)
        engine.say("Ignition of Gaulem. Loading the database.")
        engine.runAndWait()
        if not con.is_subscribed(x):
            con.subscribe_market_data(x)

        # Loading the saved model of MLP Classifier
        savename = 'Save_'+_ticker+'_'+_period+'.sav'
        _model = joblib.load(savename)
        # Loading all the stuff got from HDD added with previous version
        df_all = joblib.load(_ticker+'_'+_period)
        df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
        hourly_all = joblib.load(_ticker+'_H1')
        hourly_all = hourly_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
        df_all['WE'] = np.where(((df_all.index.weekday == 5) | (df_all.index.weekday == 6)),None,df_all.index.weekday)
        df_all = df_all.dropna()
        df_all = df_all.drop(['WE'],axis=1)
        hourly_all['WE'] = np.where(((hourly_all.index.weekday == 5) | (hourly_all.index.weekday == 6)),None,hourly_all.index.weekday)
        hourly_all = hourly_all.dropna()
        hourly_all = hourly_all.drop(['WE'],axis=1)

        print('\r.....................Base Chargée.',end='',flush=True)

        engine.say("Databases are loaded. Ready to enter Live")
        engine.runAndWait()

        # Entering the endless loop    
        while True:
            try:
                engine.say("Bulding dataframe")
                engine.runAndWait()
                print('\nConstruction de la base...')
                ##########
                
                print(x)
                if len(con.get_open_positions()) == 0:
                    if _poz==1:
                        try:
                            print(col.Back.BLACK,col.Fore.LIGHTWHITE_EX,'Position pour',x,' ouverte à',con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].open.values[-1],\
                                'et cloturée à',con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].close.values[-1],col.Style.RESET_ALL)
                        except:
                            print('Pas de position fermée détectée')
                    _poz = 0
                    joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                    print('Pas de position ouverte pour le moment')

                elif x in set(con.get_open_positions().currency.to_list()):
                    _poz = 1
                    joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                    try:
                        _tradeID = con.get_open_positions()[con.get_open_positions().currency==x].tradeId.values[-1]
                        print('TradeID =',_tradeID)
                    except:
                        _tradeID = '(non récupéré)'

                    try:
                        _amount = int(con.get_open_positions()[con.get_open_positions().currency==x].amountK.values[-1])
                    except:
                        _amount = '(non récupéré)'

                    print(col.Back.YELLOW,col.Fore.BLACK,'/!\ ATTENTION',x,' EST DEJA EN POSITION POUR UN AMOUNT DE  /!\ ',_amount,col.Style.RESET_ALL)
                    
                    try:
                        if con.get_summary()[con.get_summary().currency==x].avgBuy.values[-1] > 0:
                            _sens = ' Long'
                        elif con.get_summary()[con.get_summary().currency==x].avgSell.values[-1] > 0:
                            _sens = ' Short'
                        _pnl = con.get_summary()[con.get_summary().currency==x].grossPL.values[-1]
                        if _pnl > _pnl_max : 
                            _pnl_max = _pnl
                            joblib.dump(_pnl_max,x.replace('/','')+'_PNLMAX.dag')
                        if _pnl_min == 0:
                            _pnl_min = _pnl
                            joblib.dump(_pnl_min,x.replace('/','')+'_PNLMIN.dag')
                        if _pnl_max == 0:
                            _pnl_max = _pnl
                            joblib.dump(_pnl_max,x.replace('/','')+'_PNLMAX.dag')
                        if _pnl < _pnl_min:
                            _pnl_min = _pnl
                            joblib.dump(_pnl_min,x.replace('/','')+'_PNLMIN.dag')
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
                        _open_val = con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].open.values[-1]
                        _close_val = con.get_closed_positions()[con.get_closed_positions().tradeId==_tradeID].close.values[-1]
                        if _sens == 'Short':
                            _pnl_val = _amount  * (_open_val - _close_val) * 1000 * df_ratefx.loc[x[:3],'rate'] / 1.21
                        elif _sens == 'Long':
                            _pnl_val = _amount  * (_close_val - _open_val) * 1000 * df_ratefx.loc[x[:3],'rate'] / 1.21
                        else : 
                            _pnl_val = 'Non disponible'
                        if _pnl_val > 0:
                            print(col.Back.BLACK,col.Fore.LIGHTWHITE_EX,'Position',_sens ,'pour',x,' ouverte à',_open_val,\
                                                'et cloturée à',_close_val,'. P&L généré :',col.Fore.LIGHTGREEN_EX,_pnl_val,col.Style.RESET_ALL)
                            client.chat_postMessage(channel='log', text='Position closed '+_ticker+' horodaté à '+str(dt.datetime.now()))
                        elif _pnl_val < 0:
                            print(col.Back.BLACK,col.Fore.LIGHTWHITE_EX,'Position',_sens ,'pour',x,' ouverte à',_open_val,\
                                                'et cloturée à',_close_val,'. P&L généré :',col.Fore.LIGHTRED_EX,_pnl_val,col.Style.RESET_ALL)

                            client.chat_postMessage(channel='log', text='Position closed '+_ticker+' horodaté à '+str(dt.datetime.now()))
                        else : 
                            print(col.Back.BLACK,col.Fore.LIGHTWHITE_EX,'Position',_sens ,'pour',x,' ouverte à',_open_val,\
                                                'et cloturée à',_close_val,'. P&L généré :',col.Fore.LIGHTYELLOW_EX,_pnl_val,col.Style.RESET_ALL)

                            client.chat_postMessage(channel='log', text='Position closed '+_ticker+' horodaté à '+str(dt.datetime.now()))

                    _poz = 0
                    joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                    print('Pas de position ouverte pour',x)
                
                print('\nWaiting for the candle...')
                engine.say("Sniffing foot prints")
                engine.runAndWait()
                print()
                
                _spread = ((con.get_prices(x).Ask[-1] - con.get_prices(x).Bid[-1])/con.get_prices(x).Bid[-1])*100
                print('\nTicker tracké :',x,' ',dt.datetime.now(), ' ---------     Spread en % :',_spread,' ------ ')
                ##########
                # Wait until the minute is in the 'm5' list
                while dt.datetime.now().minute not in [0,5,10,15,20,25,30,35,40,45,50,55]:
                    print( '\rTicker tracké :',x,' ',dt.datetime.now(), ' ', end='', flush=True )
                    time.sleep(1)
                print('\nGo!')
                try:
                    _spread = ((con.get_prices(x).Ask[-1] - con.get_prices(x).Bid[-1])/con.get_prices(x).Bid[-1])*100
                    print('\nTicker tracké :',x,' ',dt.datetime.now(), ' ---------     Spread en % :',_spread,' ------ \n')
                except:
                    print('/!\ PROBLEME DE RECUPERATION DU SPREAD /!\ ')
                time.sleep(15)
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
                addon['WE'] = np.where(((addon.index.weekday == 5) | (addon.index.weekday == 6)),None,addon.index.weekday)
                addon = addon.dropna()
                addon = addon.drop(['WE'],axis=1)
                df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
                # Concatenate the bases
                df_all = df_all.append(addon.iloc[1:,:])
                #df_all = df_all.iloc[-263570:,:]
                if dt.datetime.now().minute - df_all.index[-1].minute < 5:
                    df_all = df_all.iloc[:-1,:]
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
                hourly_add['WE'] = np.where(((hourly_add.index.weekday == 5) | (hourly_add.index.weekday == 6)),None,hourly_add.index.weekday)
                hourly_add = hourly_add.dropna()
                hourly_add = hourly_add.drop(['WE'],axis=1)
                hourly_all = hourly_all.append(hourly_add.iloc[1:,:])
                if dt.datetime.now().hour - hourly_add.index[-1].hour == 1:
                    hourly_add = hourly_add.iloc[:-1,:]
                hourly_all = timerange1D(hourly_all)

                # Make daily & weekly bases
                daily_all = get_daily(hourly_all)
                daily_all = timerange1W(daily_all)
                weekly_all = get_weekly(daily_all)

                # Calculate the indicators
                daily_all = adr(daily_all,_window=14)
                df_all = getadr(daily_all,df_all)
                df_all = adrhnl(df_all)
                df_all = sma(df_all=df_all,_window=200)
                df_all = bollinger(df_all,_slow=20)
                df_all = slowstochastic(df_all)
                df_all = ema(df_all,21)
                df_all = ema(df_all,8)

                weekly_all = pivot(weekly_all)
                df_all = pivotimportdf(df_all,weekly_all)
                df_all = atr(df_all,14)
                df_all = rvi(df_all,_window=14)
                df_all = sbgamma(df_all)
                df_all = onhisma(df_all,_window=5)
                df_all = onlosma(df_all,_window=5)
                df_all = onhisma(df_all,_window=21)
                df_all = onlosma(df_all,_window=21)
                df_all = onhisma(df_all,_window=34)
                df_all = onlosma(df_all,_window=34)
                df_all = importohlc(df_all,weekly_all,_suffix='_weekly')
                df_all = importohlc(df_all=df_all,other_all=daily_all,_suffix='_daily')
                
                # Apply the raw strategy to get the signals
                df_all = stochastic(df_all)
                
                # Featuring
                features = featuring(df_all)

                # And drop the nan
                features = features.dropna()

                for label in features.drop(['Date','Symbol','Signal'],axis=1).columns:
                    features[label] = features[label] / features_max[label][0]

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

                    print(col.Fore.MAGENTA,'\nTest sur la bougie',features.index[-1],col.Style.RESET_ALL)


                    if _valid == 1 and _signal == 1 and _poz == 0 :
                        if _spread <= _trigger_spread :
                            try:
                                buy(_period)
                                _poz = 1
                                joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                                client.chat_postMessage(channel='log', text='Open Long '+_ticker+' horodaté à '+str(dt.datetime.now()))
                            except:
                                print("L'ordre n'a pas été exécuté")
                                client.chat_postMessage(channel='log', text="L'ordre n'a pas pu être executé, "+_ticker+" horodaté à "+str(dt.datetime.now()))
                                _poz = 0
                                joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                        else:
                            print(col.Fore.BLACK,'SPREAD FAILED!!!!!!!!',col.Style.RESET_ALL)
                            client.chat_postMessage(channel='log', text='Spread Failure on Long '+_ticker+' horodaté à '+str(dt.datetime.now()))
                            _poz = 0
                            joblib.dump(_poz,x.replace('/','')+'_POZ.dag')

                    elif _valid == 1 and _signal == -1 and _poz == 0 :
                        if _spread <= _trigger_spread :
                            try:
                                sell(_period)
                                _poz = 1
                                joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                                client.chat_postMessage(channel='log', text='Open Short '+_ticker+' horodaté à '+str(dt.datetime.now()))
                            except:
                                print("L'ordre n'a pas été exécuté")
                                client.chat_postMessage(channel='log', text="L'ordre n'a pas pu être executé, "+_ticker+" horodaté à "+str(dt.datetime.now()))
                                _poz = 0
                                joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                        else:
                            print(col.Fore.BLACK,'SPREAD FAILED!!!!!!!!',col.Style.RESET_ALL)
                            client.chat_postMessage(channel='log', text='Spread Failure Shirt '+_ticker+' horodaté à '+str(dt.datetime.now()))
                            _poz = 0
                            joblib.dump(_poz,x.replace('/','')+'_POZ.dag')
                        

                    elif _valid == 1 and _signal == 1 and _poz != 0 :
                        con.close_trade(trade_id=int(_tradeID),amount=_amount)
                        print(col.Back.LIGHTRED_EX,col.Fore.LIGHTCYAN_EX,'/!\ INVERSE CLOSE INVOKED /!\ ',col.Style.RESET_ALL)
                        client.chat_postMessage(channel='log', text=('INVERSE CLOSE '+_ticker+' horodaté à '+str(dt.datetime.now())))
                    else:
                        print(col.Fore.BLUE,'\nNO SIGNAL FOR',col.Fore.YELLOW,x,'\n',col.Style.RESET_ALL)
                        print('Raw signal to',_signal)
                        print(df_all[['slow_K5','slow_D5']].tail(2))
                
                print('Sauvegarde des bases.')

                # Save the base
                joblib.dump(df_all , _ticker+'_'+_period)
                joblib.dump(hourly_all , _ticker+'_H1')

                print('Reset des bases')
                # Purification
                df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
                hourly_all = hourly_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
            
            except:
                print(col.Back.BLACK,col.Fore.YELLOW,'/!\ EXCEPTION RAISED - TRYING TO LOOP AGAIN -- /!\ ')
                print("Analyse des procédures d'urgence")
                engine.say("Emergency situation raised. Incoming Crash. Trying to fix it")
                engine.runAndWait()
                print('Heure du crash :',dt.datetime.now())
                print('derniere bougie relevée :')
                print()
                print(df_all.iloc[-1,:])
                print()
                client.chat_postMessage(channel='log', text='Emergency Procedure raised '+_ticker+' horodaté à '+str(dt.datetime.now()))
                try:
                    print('Raw Signal de la dernière bougie avant le crash', _signal)
                except:
                    pass
                print()
                
                while dt.datetime.now().minute in [0,5,10,15,20,25,30,35,40,45,50,55]:
                    print('\rAttente pour relancer le moteur :',x,' ',dt.datetime.now(),end='',flush=True)
                    time.sleep(0.97)
                print('\nRéamorçage du moteur\n!')
                try :
                    if con.is_connected() == True:
                        print('Golem est toujours connecté')
                    else:
                        print('Golem est déconnecté. Tentative de reconnexion en cours')
                        import sys
                        a_del=[]
                        for module in sys.modules.keys():
                            if 'fxcm' in module:
                                a_del.append(module)

                        for module in a_del:
                            del sys.modules[module]

                        del fxcmpy
                        import fxcmpy
                        
                        con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
                        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
                        print('Compte utilisé : ',con.get_account_ids())
                        engine.say("Gaulem is Connected")
                        engine.runAndWait()
                        print('\n Moteur amorcé...')
                        engine.say("Ignition of Gaulem. Trying to go back.")
                        engine.runAndWait()
                        print(col.Style.RESET_ALL)
                        client.chat_postMessage(channel='log', text='Reconnexion '+_ticker+' horodaté à '+str(dt.datetime.now()))
                        break
                except:
                    engine.say("Mayday! Mayday! Nothing could be done.")
                    engine.runAndWait()
                    client.chat_postMessage(channel='log', text='Mayday! Mayday --- Arrêt du système exigé ---'+_ticker+' horodaté à '+str(dt.datetime.now()))
                    print('Arrêt du système à ',dt.datetime.now())
                    print(col.Style.RESET_ALL)
                    break
                print('trying to loop again')
                break
                

if __name__ == "__main__":
    pass
