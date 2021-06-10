import pandas as pd
import numpy as np
import datetime as dt
import talib
import os
import sys
sys.path.append('../')

def baseimport(x,_period,_year_bottom,_year_top):
    
    base = pd.read_csv('/Volumes/DATA_SCIENCES/DEV_EN_COURS/Bases/Base_Clean/'+x.replace('/','')+_period+'.csv')

    ##### On fixe la date en index sous forme de Timestamp
    base.set_index(pd.to_datetime(base.Date),drop=True,inplace=True)

    ###### On drop les colonnes inutiles
    base = base.drop(['Date','Total'],axis=1)

    ##### On enlève les jours correspondant au samedi et au dimanche
    base['WE'] = np.where(((base.index.weekday == 5) | (base.index.weekday == 6)),None,base.index.weekday)
    base = base.dropna()
    base = base.drop(['WE'],axis=1)
    print('Base chargée')
    print('\nBorne inférieure de la base:',base.index[0],', Borne supérieure :',base.index[-1])

    # Copie de sécurité de la base pour manioulation ultérueure, et restriction à la période

    df = base[(base.index>=_year_bottom)&(base.index<=_year_top)]
    print('\nBase réduite à df, ayant pour shape :',df.shape)
    print('\nBorne inférieure de df:',df.index[0],', Borne supérieure de df:',df.index[-1])
    print('')
    return(base,df.sort_index(axis=1))

def get_all_data(_period,TICKER_LIST):
    df_all = pd.DataFrame()
    df_temp = pd.DataFrame()
    for ticker in TICKER_LIST:
        df_temp = pd.read_csv('../BASES/Base/'+ticker.replace('/','')+'_'+_period+'_BidAndAsk.csv')
        df_temp['Symbol'] = ticker
        df_all = df_all.append(df_temp)
    return(df_all)

def get_ticker_list():
    KEY_LIST = []
    RAW_LIST = os.listdir('../BASES/Base/')
    KEY_LIST = [(_cur.split('_')[0][:3]+'/'+_cur.split('_')[0][3:]) for _cur in RAW_LIST]
    return(KEY_LIST)

def jyss_oscillator(df):
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

    df['HiWin'] = df.High.rolling(10).max()
    df['LoWin'] = df.Low.rolling(10).min()
    df['JyssOscBear'] = ((df.HiWin-df.HiWin.shift(2))/df.HiWin.shift(2))*1000
    df['JyssOscBearSD'] = df.JyssOscBear.rolling(20).std()
    df['JyssOscBull'] = ((df.LoWin-df.LoWin.shift(2))/df.LoWin.shift(2))*1000
    df['JyssOscBullSD'] = df.JyssOscBull.rolling(20).std()
    df['TriggerBear'] = (df.HiWin - df.HiWin.shift(2)) * 1000
    df['TriggerBull'] = (df.LoWin - df.LoWin.shift(2)) * 1000
    df['Trigger'] = np.where((df.LoWin - df.LoWin.shift(9) == 0),1,np.where((df.HiWin - df.HiWin.shift(9) == 0 ),1,0))
    '''df['SigJyssOsc'] = np.where(\
    ((df.Close <= 1.01*df.LoWin)&(df.JyssOscBull <= -4*df.JyssOscSD)),1,\
        np.where(\
            ((df.Close >= 0.99*df.HiWin)&(df.JyssOscDwn >= 4*df.JyssOscDwnSD)),-1,0)
)'''
    return(df.sort_index(axis=1))

def rsi(df,_tp):
    # Renvoit le RSI avec en indice la time_period
    df['RSI_'+str(_tp)] = talib.RSI(df.Close,timeperiod = _tp)
    return(df.sort_index(axis=1))

def aroon(df,_tp):
    ar_up,ar_low = talib.AROON(df.High,df.Low,_tp)
    df['AROON_UP'] = ar_up
    df['AROON_LOW'] = ar_low
    df['AROON'] = ar_up - ar_low
    return(df.sort_index(axis=1))

def bollinger(df,_tp):
    upper, middle, lower = talib.BBANDS(df.Close, timeperiod=_tp, nbdevup=2, nbdevdn=2)
    df['BOL_UP_'+str(_tp)] = upper
    df['Bol_LO_'+str(_tp)] = lower
    df['SB_VOL_'+str(_tp)] = upper - lower
    return(df.sort_index(axis=1))

def momentum(df,_tp):
    df['MOMENTUM_'+str(_tp)] = talib.MOM(df.Close, timeperiod=_tp)
    return(df.sort_index(axis=1))

def tsf(df,_tp):
    df['TSF'] = talib.TSF(df.Close,timeperiod=_tp)
    return(df.sort_index(axis=1))

def lineregangle(df,_tp):
    df['LINEAR_REGR_ANGLE'] = talib.LINEARREG_ANGLE(df.Close,timeperiod=_tp)
    return(df.sort_index(axis=1))

def linereg(df,_tp):
    df['LINEAR_REGR'] = talib.LINEARREG(df.Close,timeperiod=_tp)
    return(df.sort_index(axis=1))

def lineregslope(df,_tp):
    df['LINEAR_REGR_SLOPE'] = talib.LINEARREG_SLOPE(df.Close,timeperiod=_tp)
    return(df.sort_index(axis=1))

def mma(df,_tp):
    df['MMA_'+str(_tp)] = talib.EMA(df.Close,timeperiod=_tp)
    return(df.sort_index(axis=1))

def sma(df,_tp):
    df['SMA_'+str(_tp)] = talib.SMA(df.Close,timeperiod=_tp)
    return(df.sort_index(axis=1))

def atr(df,_tp):
    df['ATR_'+str(_tp)] = talib.ATR(df.High, df.Low, df.Close, timeperiod=_tp)
    return(df.sort_index(axis=1))

def stochastic(df,_fp,_sp):
    slowk, slowd = talib.STOCH(df.High, df.Low, df.Close, fastk_period=_fp, slowk_period=_sp, slowk_matype=0, slowd_period=_sp, slowd_matype=0)#STOCHASTIC
    df['STOC_SLOWK'] = slowk
    df['STOC_SLOWD'] = slowd
    return(df.sort_index(axis=1))

def stochastic_rsi(df,_tp,_fp,_so):
    fastk, fastd = talib.STOCHRSI(df.Close, timeperiod=_tp, fastk_period=_fp, fastd_period=_fp, fastd_matype=0)#STOCHASTICRSI
    df['STOCRSI_FASTK'] = fastk
    df['STOCHRSI_FASTD'] = fastd
    return(df.sort_index(axis=1))

def macd(df,_fp,_sp,_tp):
    macd, macdsignal, macdhist = talib.MACD(df.Close, fastperiod=_fp, slowperiod=_sp, signalperiod=_tp)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macdsignal
    df['MACD_HIST'] = macdhist
    return(df.sort_index(axis=1))

def dx(df,_tp):
    df['DIRECTIONAL_INDEX'] = talib.DX(df.High,df.Low,df.Close,_tp)
    return(df.sort_index(axis=1))

def atr(df,_tp):
    df['ATR'] = talib.ATR(df.High,df.Low,df.Close,_tp)
    return(df.sort_index(axis=1))


if __name__ == "__main__":
    pass 