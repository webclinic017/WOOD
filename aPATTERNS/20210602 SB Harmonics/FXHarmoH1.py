__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'
__version__ = 'v1.0'

####################################################################
####################################################################
########################### GOLEM HAROMNICS ########################
####################################################################
####################################################################

'''
In this version, Go!em is scanning the market and is looking for harmonics. If found, it sends them on its telegram
'''

## Librairies
import datetime as dt
import time
import joblib
import sys
import fxcmpy
import colorama as col
import pandas as pd
import numpy as np
from tqdm import tqdm
import talib
from scipy.signal import argrelextrema
import mplfinance as fplt
import requests
import json
import os
sys.path.append('../') 

## Paramètres
TICKER_LIST = ['EUR/USD','USD/JPY','GBP/USD','USD/CHF','EUR/CHF','AUD/USD','USD/CAD','NZD/USD','EUR/GBP','EUR/JPY','GBP/JPY','CHF/JPY','GBP/CHF','EUR/AUD','EUR/CAD',\
    'AUD/CAD','AUD/JPY','CAD/JPY','NZD/JPY','GBP/CAD','GBP/NZD','GBP/AUD','AUD/NZD','USD/SEK','EUR/SEK','EUR/NOK','USD/NOK','USD/MXN','AUD/CHF','EUR/NZD','USD/ZAR','ZAR/JPY',\
        'NZD/CHF','CAD/CHF','NZD/CAD','USD/CNH','US30','WHEATF','XAU/USD','XAG/USD']

err_allowed = 10/100
_period = 'H1'
_token = joblib.load('TOKENS/_api_token.dag')
_bot_token = joblib.load('TOKENS/telegram_token.dag')
_chat_id = joblib.load('TOKENS/telegram_chat_id.dag')
_server = 'demo'
_number = 500
_lookback = 1
_con = None
## Fontions

# Envoi message texte
def telegram_message(_message):
    message = 'https://api.telegram.org/bot'+ _bot_token +  '/sendMessage?chat_id=' + _chat_id+ '&parse_mode=Markdown&text=' + _message
    send = requests.post(message)

# Envoi image
def telegram_pic(_pic):
    files = {'photo': open(_pic, 'rb')}
    message = ('https://api.telegram.org/bot'+ _bot_token + '/sendPhoto?chat_id=' + _chat_id)
    send = requests.post(message, files = files)

# Réduit la dataframe à OHLC
def reduce_df(df):
    df = df[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
    return(df)

# Vérifie la présence de we
def is_we(dataframe_to_check):
    IDX = dataframe_to_check.index.to_list()
    c=0
    for day in IDX:
        if day.weekday() == 5 or day.weekday() == 6:
            c += 1
    print('Nombre de samedi et dimanches présents :',c)

# Drop les lignes appartenant à un we
def drop_we(df):
    df['WE'] = np.where(((df.index.weekday == 5) | (df.index.weekday == 6)),None,df.index.weekday)
    df = df.dropna()
    df = df.drop(['WE'],axis=1)
    return(df)

# Met à jour la base
def maj(df, con,_tf,_decay=0):
    
    _fin = dt.datetime.now()
    _deb = df.index[-1]
    _debut = dt.datetime(_deb.year,_deb.month,_deb.day,_deb.hour,_deb.minute)
    # Scrap the addon & build it to be compliant with our df
    addon = get_candl(con,_debut, _fin,_tf)
    # Calculate the mid prices

    addon = make_mid(addon)
    addon = drop_we(addon)
    addon = make_mid(addon)
    addon['Symbol'] = _ticker
    addon['Date'] = addon.index
    addon['Date'] = pd.to_datetime(addon['Date'].dt.strftime(date_format='%Y-%m-%d'))

    # Concatenate the bases
    df = df.append(addon.iloc[1:,:])
    #df = df.iloc[-263570:,:]
    if _tf == 'm5':
        if (dt.datetime.now().minute - df.index[-1].minute <= 5) : # or ((dt.datetime.now().minute - df.index[-1].minute == -55)) :
            df = df.iloc[:-1,:]
            print('Cut en m5 - Dernière bougie _period récupérée :',df.index[-1])
        else :
            print('No Cut en m5 - Dernière bougie _period récupérée :',df.index[-1])

    elif _tf == 'm15':
        if (dt.datetime.now().minute - df.index[-1].minute <= 15) : # or ((dt.datetime.now().minute - df.index[-1].minute == -55)) :
            df = df.iloc[:-1,:]
            print('Cut en m15 - Dernière bougie _period récupérée :',df.index[-1])
        else :
            print('No Cut en m15 - Dernière bougie _period récupérée :',df.index[-1])

    elif _tf == 'm30':
        if (dt.datetime.now().minute - df.index[-1].minute <= 30) : # or ((dt.datetime.now().minute - df.index[-1].minute == -55)) :
            df = df.iloc[:-1,:]
            print('Cut en m30 - Dernière bougie _period récupérée :',df.index[-1])
        else :
            print('No Cut en m30 - Dernière bougie _period récupérée :',df.index[-1])
    
    if _tf == 'H1' :
        if dt.datetime.now().hour - df.index[-1].hour == _decay :
            df = df.iloc[:-1,:]
            print('Cut en H1 - Dernière bougie _period récupérée :',df.index[-1])
    
    if _tf == 'H4' :
        if dt.datetime.now().hour - df.index[-1].hour == 4 + _decay :
            df = df.iloc[:-1,:]
            print('Cut en H4 - Dernière bougie _period récupérée :',df.index[-1])
    
        else :
            print('No Cut en H4 - Dernière bougie _period récupérée :',df.index[-1])
    return(df)

# Calcul les valeurs mid à partir des Bid/Ask
def make_mid(df):
    df['Open'] = (df.OpenAsk + df.OpenBid)/2
    df['High'] = (df.HighAsk + df.HighBid)/2
    df['Low'] = (df.LowAsk + df.LowBid)/2
    df['Close'] = (df.CloseAsk + df.CloseBid)/2
    df['Symbol'] = _ticker
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'].dt.strftime(date_format='%Y-%m-%d'))
    df = drop_we(df)
    return(df)

# Récupère les boudies fxcm
def get_candl(con,x,_tf,_number=200):
    df = con.get_candles(x,period=_tf,number=_number).drop(['tickqty'],axis=1)
    df = df.rename(columns={'bidopen':'OpenBid','bidclose':'CloseBid','bidhigh':'HighBid','bidlow':'LowBid','askopen':'OpenAsk','askclose':'CloseAsk','askhigh':'HighAsk','asklow':'LowAsk'})
    return(df)

# Connexion à fxcm
def conX(con,_token,_server):
    #global con
    try:
        con.is_connected()
        if con.is_connected() == True:
            print('Déjà connecté')
            print('')
        else:
            con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
            print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
            print('Compte utilisé : ',con.get_account_ids())
            print('')
    except:
        con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
        if con.is_connected() == True:
            print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
            print('Compte utilisé : ',con.get_account_ids())
            print('')
        else:
            print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
            print('')
    return(con)

# Detection des peaks / valleys
def peak_detect(high,low,order=10):
    #print('price.shape',price.shape)
    max_idx = list(argrelextrema(high,np.greater,order=order)[0])
    #print('max_idx shape',len(max_idx))
    min_idx = list(argrelextrema(low,np.less,order=order)[0])
    #print('min_idx shape',len(min_idx))
    idx = max_idx + min_idx
    #print('idx shape',len(idx))
    idx.sort()
    CURRENT = []
    for i in idx:
        if i in max_idx:
            CURRENT.append(high[i])
        else:
            CURRENT.append(low[i])


    if idx[-1] in max_idx:
        CURRENT.append(low[len(high)-1])

    else:
        CURRENT.append(high[len(low)-1])
    
    CURRENT = CURRENT[-5:]
    _current_idx = idx[-4:] + [len(high)-1]
    #print('current_idx shape',len(current_idx))
    _start = min(_current_idx)
    _end = max(_current_idx)
    return _current_idx,CURRENT,_start,_end 

# Detection des Gartley
def is_gartley(moves,err_allowed):  
    XA=moves[0]
    AB=moves[1]
    BC=moves[2]
    CD=moves[3]
   
    AB_range = np.array([0.618 - err_allowed,0.618 + err_allowed])*abs(XA)
    BC_range = np.array([0.382 - err_allowed,0.886 + err_allowed])*abs(AB)
    CD_range = np.array([1.27 - err_allowed,1.618 + err_allowed])*abs(BC)
        
    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return 1
        else:
            return np.isnan
        
    elif XA<0 and AB>0 and BC<0 and CD>0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return -1
        else:
            return np.isnan
    else:
        return np.isnan

# Detection des Butterfly
def is_butterfly(moves,err_allowed):   
    XA=moves[0]
    AB=moves[1]
    BC=moves[2]
    CD=moves[3]
    
    AB_range = np.array([0.786 - err_allowed,0.786 + err_allowed])*abs(XA)
    BC_range = np.array([0.382 - err_allowed,0.886 + err_allowed])*abs(AB)
    CD_range = np.array([1.618 - err_allowed,2.618 + err_allowed])*abs(BC)
        
    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return 1
        else:
            return np.NaN
        
    elif XA<0 and AB>0 and BC<0 and CD>0:      
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return -1
        else:
            return np.isnan
    else:
        return np.isnan       

# Detection des Crab
def is_crab(moves,err_allowed):      
    XA=moves[0]
    AB=moves[1]
    BC=moves[2]
    CD=moves[3]
    
    AB_range = np.array([0.382 - err_allowed,0.618 + err_allowed])*abs(XA)
    BC_range = np.array([0.382 - err_allowed,0.886 + err_allowed])*abs(AB)
    CD_range = np.array([2.24 - err_allowed,3.618 + err_allowed])*abs(BC)
           
    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return 1
        else:
            return np.NaN
        
    elif XA<0 and AB>0 and BC<0 and CD>0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return -1
        else:
            return np.isnan
    else:
        return np.isnan      

# Detection des Bat
def is_bat(moves,err_allowed):  
    XA=moves[0]
    AB=moves[1]
    BC=moves[2]
    CD=moves[3]
    
    AB_range = np.array([0.382 - err_allowed,0.5 + err_allowed])*abs(XA)
    BC_range = np.array([0.382 - err_allowed,0.886 + err_allowed])*abs(AB)
    CD_range = np.array([1.618 - err_allowed,2.618 + err_allowed])*abs(BC)
    
    if XA>0 and AB<0 and BC>0 and CD<0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return 1
        else:
            return np.NaN
        
    elif XA<0 and AB>0 and BC<0 and CD>0:
        if AB_range[0]<abs(AB)<AB_range[1] and BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]:
            return -1
        else:
            return np.isnan
    else:
        return np.isnan
    
# Detection des ABCD
def is_abcd(moves,err_allowed):  
    AB=moves[1]
    BC=moves[2]
    CD=moves[3]
    
    BC_range = np.array([0.618 - err_allowed,0.618 + err_allowed])*abs(AB)
    CD_range = np.array([1.618 - err_allowed,1.618 + err_allowed])*abs(BC)
    CD2_range = np.array([1 - err_allowed,1 + err_allowed])*abs(AB)
    
    if AB<0 and BC>0 and CD<0 :    
        if BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1] \
        and CD2_range[0]<abs(CD)<CD2_range[1] :
            return 1
        else:
            return np.NaN
        
    elif AB>0 and BC<0 and CD>0 :
        if BC_range[0]<abs(AB)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1]\
        and CD2_range[0]<abs(CD)<CD2_range[1]:
            return -1
        else:
            return np.isnan
    else:
        return np.isnan    

# Detection des Shark
def is_shark(moves,err_allowed):  
    
    XA=moves[0]
    AB=moves[1]
    BC=moves[2]
    CD=moves[3]
  
    #CD_range = np.array([0.886 - err_allowed,1.13 + err_allowed])*abs(XB)
    BC_range = np.array([1.13 - err_allowed,1.618 + err_allowed])*abs(AB)
    CD_range = np.array([1.618 - err_allowed,2.24 + err_allowed])*abs(BC)
    CD_range2 = np.array([0.88 - err_allowed,1.13 + err_allowed])*abs(XA)   
    
    if XA>0 and AB<0 and BC>0 and CD<0 :
        if BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1] and CD_range2[0]<abs(CD)<CD_range2[1]:
            return 1
        else:
            return np.NaN
        
    elif XA<0 and AB>0 and BC<0 and CD>0 :
        if  BC_range[0]<abs(BC)<BC_range[1] and CD_range[0]<abs(CD)<CD_range[1] and CD_range2[0]<abs(CD)<CD_range2[1]:
            return -1
        else:
            return np.isnan
    else:
        return np.isnan


if __name__ == "__main__":
    print('Global Optimized LumberJack Environment Motor for For_Ex\nLumberJack Jyss 5781(c)')
    print(col.Fore.CYAN,'°0Oo_D.A.G._26_oO0°')
    print(col.Fore.YELLOW,col.Back.BLUE,'--- Golem Harmo FXCM 1H #v1.0 ---',col.Style.RESET_ALL)
    print('')
    con = None
    try:
        con
    except NameError:
        con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
                
    else:
        con = conX(con,_token,_server)


    try:   
        _last_bougie = (con.get_candles('EUR/USD',period='m5',number=1).index.hour[0])
        _now = dt.datetime.now().hour

        if _now - _last_bougie < 0:
            _decay = _now + 24 - _last_bougie

        else:
            _decay = _now - _last_bougie


        print('\nDecay = ',_decay,'\n')
        joblib.dump(_decay,'VARS/decay.dag')
        _no_access = 0
    except:
        print('\n',col.Back.RED,col.Fore.BLACK,'/!\ LES DONNEES NE SONT PAS ACCESSIBLES. PAS DE MAJ NI DE LIVE /!\ ',col.Style.RESET_ALL)
        time.sleep(0.1)
        _no_access = 1

while True:
    while dt.datetime.now().minute != 0:
        print('\rAttente de la nouvelle bougie - Mise en veille - Heure locale :',dt.datetime.now(),end='',flush=True)
        time.sleep(1)
        
   
    print('Scan, heure locale :',dt.datetime.now())

    _last_bougie = (con.get_candles('EUR/USD',period='m5',number=1).index.hour[0])
    _now = dt.datetime.now().hour

    if _now - _last_bougie < 0:
        _decay = _now + 24 - _last_bougie
    else:
        _decay = _now - _last_bougie
    
    #TICKER_LIST = con.get_instruments()
    #for t in ['US.BANKS','ACA.fr','AI.fr','AIR.fr','ORA.fr','MC.fr','RNO.fr','BAYN.de','BMW.de','DPW.de','DTE.de','AZN.uk','BP.uk','GSK.uk','TSCO.uk','RDSB.uk']:
    #    TICKER_LIST.remove(t)

    BAD_TICKERS = []
    _compteur = 0
    # TICKER_LIST = con.get_instruments()
    for x in tqdm(TICKER_LIST):
        #try:
        _ticker = x.replace('/','')
        df = get_candl(con,x,_period,_number)
        if _period == 'm5':
            if (dt.datetime.now().minute - df.index[-1].minute <= 5) : # or ((dt.datetime.now().minute - df.index[-1].minute == -55)) :
                df = df.iloc[:-1,:]
                #print('\rCut en m5 - Dernière bougie _period récupérée :',df.index[-1],end='')
            else :
                #print('\rNo Cut en m5 - Dernière bougie _period récupérée :',df.index[-1],end='')
                pass

        elif _period == 'm15':
            if (dt.datetime.now().minute - df.index[-1].minute <= 15) : # or ((dt.datetime.now().minute - df.index[-1].minute == -55)) :
                df = df.iloc[:-1,:]
                #print('\rCut en m15 - Dernière bougie _period récupérée :',df.index[-1],end='')
            else :
                #print('\rNo Cut en m15 - Dernière bougie _period récupérée :',df.index[-1],end='')
                pass

        elif _period == 'm30':
            if (dt.datetime.now().minute - df.index[-1].minute <= 30) : # or ((dt.datetime.now().minute - df.index[-1].minute == -55)) :
                df = df.iloc[:-1,:]
                #print('\rCut en m30 - Dernière bougie _period récupérée :',df.index[-1],end='')
            else :
                #print('\rNo Cut en m30 - Dernière bougie _period récupérée :',df.index[-1],end='')
                pass
        
        if _period == 'H1' :
            time.sleep(60)
            if dt.datetime.now().hour - df.index[-1].hour == _decay :
                df = df.iloc[:-1,:]
                #print('\rCut en H1 - Dernière bougie _period récupérée :',df.index[-1],end='')
            else :
                #print('\rNo Cut en H4 - Dernière bougie _period récupérée :',df.index[-1],end='')
                pass
        
        if _period == 'H4' :
            time.sleep(90)
            if dt.datetime.now().hour - df.index[-1].hour == _decay :
                df = df.iloc[:-1,:]
                #print('\rCut en H4 - Dernière bougie _period récupérée :',df.index[-1],end='')
        
            else :
                #print('\rNo Cut en H4 - Dernière bougie _period récupérée :',df.index[-1],end='')
                pass

        df = drop_we(df)
        df = make_mid(df) 

        price = df['Close']
        high = df['High']
        low = df['Low']
        rsi = talib.RSI(price, timeperiod=14)

        
        i = len(price) - 1

        # current_idx,current_pat,start,end = peak_detect(price.values[:i+1],low.values[:i+1],high.values[:i+1])
        current_idx,current_pat,start,end = peak_detect(high.values[:i+1],low.values[:i+1])

        XA = current_pat[1] - current_pat[0]
        AB = current_pat[2] - current_pat[1]
        BC = current_pat[3] - current_pat[2]
        CD = current_pat[4] - current_pat[3]

        moves=[XA,AB,BC,CD]

        gartley = is_gartley(moves,err_allowed)
        butterfly = is_butterfly(moves,err_allowed)
        crab = is_crab(moves,err_allowed)
        bat = is_bat(moves,err_allowed)
        shark = is_shark(moves,err_allowed)
        abcd = is_abcd(moves,err_allowed)

        current_rsi = [rsi.iloc[current_idx[0]],rsi.iloc[current_idx[1]],rsi.iloc[current_idx[2]],rsi.iloc[current_idx[3]],rsi.iloc[current_idx[4]]]
        current_rsi_max = max(current_rsi)
        current_rsi_min = min(current_rsi)
        delta_price = current_pat[4] - current_pat[2]
        delta_rsi = current_rsi[4] - current_rsi[2]

        harmonics = np.array([gartley,butterfly,bat,crab,shark,abcd])
        #harmonics = np.array([abcd])
        labels = ['gartley','butterly','bat','crab','shark','abcd'] 
        #labels = ['abcd']
        #if (np.any(harmonics==1) and delta_price<0 and delta_rsi>0) or (np.any(harmonics==-1) and delta_price>0 and delta_rsi<0):
        if np.any(harmonics==1) or np.any(harmonics==-1):
            _compteur += 1
            for j in range(0, len(harmonics)):
                _pic = str(i)+'.png'
                _directory = str(df.index[current_idx[-1]]).replace('-','_').replace(' ','_').replace(':','_')+'/'+_period+'/'
        
                if not os.path.exists('RESULTS/'+_directory):
                    os.makedirs('RESULTS/'+_directory)

                if harmonics[j]==1 or harmonics[j]==-1:
                    sense ='bearish' if harmonics[j]==-1 else 'bullish'
                    label =  sense + ' '+labels[j] +' found in TimeFrame '+_period
                    
                    _signals = [(df.index[current_idx[0]],current_pat[0]),(df.index[current_idx[1]],current_pat[1]),(df.index[current_idx[2]],current_pat[2])\
                                ,(df.index[current_idx[3]],current_pat[3]),(df.index[current_idx[4]],current_pat[4])]
                    df['Scatter'] = np.where(df.index==df.index[current_idx[-1]],current_pat[-1],np.nan)
                    if sense == 'bullish':
                        _line_price = fplt.make_addplot(df.iloc[start:,:].Scatter.to_list(),type='scatter',markersize=150,marker='^',color='green')
                    if sense == 'bearish':
                        _line_price = fplt.make_addplot(df.iloc[start:,:].Scatter.to_list(),type='scatter',markersize=150,marker='v',color='r')
                    fplt.plot(df.iloc[start:,:], type='candle',title=_ticker+' - '+label,ylabel='Price',figscale=1.1,  datetime_format='%d-%m-%Y',alines=_signals,addplot=_line_price,\
                        savefig='RESULTS/'+_directory+_pic)
                    
                    telegram_message('Ticker : '+x+'\nTimeFrame : '+_period+"\nSignal's Sense : "+sense+"\nSignal's Date : "+str(df.index[current_idx[-1]])+"\nClose's Price : "+\
                            str(round(price[i],4))+"\nScatter's Price : "+str(round(current_pat[-1],4)))
                    
                    telegram_pic('RESULTS/'+_directory+_pic)
                    time.sleep(0.1)
                    print(df.index[current_idx],df.index[i])
   
                
    if _compteur <= 1:
        print('\nIl y a eu ',_compteur,'signal de trouvé dans la TimeFrame',_period)
        telegram_message('\nIl y a eu '+str(_compteur)+' signal de trouvé dans la TimeFrame '+_period)
    else:
        print('\nIl y a eu ',_compteur,'signaux de trouvé dans la TimeFrame',_period)
        telegram_message('\nIl y a eu '+str(_compteur)+' signaux de trouvé dans la TimeFrame '+_period)

    while dt.datetime.now().minute == 0:
        print('\rAttente de la nouvelle bougie - Mise en veille - Heure locale :',dt.datetime.now(),end='',flush=True)
        time.sleep(1)