__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
####### RECUPERATION DONNEES ET CONSTRUCTION DES DATA FX ###########
####################################################################
####################################################################

import time
import sys
sys.path.append('../')
from  librairies.dagfeaturingfx import *
from tqdm import tqdm
import joblib
from joblib import Parallel,delayed
import pyttsx3
import colorama as col
engine = pyttsx3.init()
engine.say("Ignition of Base Builder Motor")
engine.runAndWait()

_period = input('Enter the timeframe')

_t1 = dt.datetime.now()
print('Début des opérations horodatée à',col.Fore.YELLOW,dt.datetime.now(),col.Style.RESET_ALL)

TICKER_LIST = get_ticker_list()

engine.say("Processing featuring of dataframes, daily and intra-day")
engine.runAndWait()
##### Récupération des data pour tous les tickers sur la période demandée en intraday => df_all
df_all = get_all_data(TICKER_LIST,_period=_period)
engine.say("Raw data are loaded in memory")
engine.runAndWait()

##### Si la période demandée est déjà H1, on peut construire directement la base daily pour tous les tickers => daily_all
if _period == 'H1':
    df_all = timerange1D(df_all)
    daily_all = get_daily(df_all,TICKER_LIST)

##### Si la période n'est pas H1, on récupère d'abord les data en 1H pour tous les tickers, et on construit la base daily à partir du 1H => daily_all
else:
    df_all = timerange1D(df_all)
    hourly_all = get_all_data(TICKER_LIST,_period='H1')
    hourly_all = timerange1D(hourly_all)
    daily_all = get_daily(hourly_all,TICKER_LIST)
    del hourly_all

engine.say("Daily up to date")
engine.runAndWait()

daily_all = timerange1W(daily_all)
weekly_all = get_weekly(daily_all,TICKER_LIST)
engine.say("Weekly up to date")
engine.runAndWait()

##### On calcule l'ADR sur le daily
daily_all = adr(daily_all,_window=14)
engine.say("ADR computed")
engine.runAndWait()

##### On récupère l'ADR qui a été calculé en daily (daily_all) pour le mettre dans la base intraday df_all
df_all = getadr(daily_all,df_all,TICKER_LIST)
engine.say("ADR get in daily")
engine.runAndWait()

time.sleep(50)

##### Calcul d'une SMA 200 sur df_all
df_all = sma(df_all=df_all,_window=200)

##### Calcul des bollinger sur df_all
df_all = bollinger(df_all,_slow=20)

##### Calcul du stochastic slow. Par défaut les paramètres sont 5 et 3.
df_all = slowstochastic(df_all,TICKER_LIST)

df_all = ema(df_all,21,TICKER_LIST)

df_all = ema(df_all,8,TICKER_LIST)

weekly_all = pivot(weekly_all,TICKER_LIST)

df_all = pivotimportdf(df_all,weekly_all,TICKER_LIST)

time.sleep(30)

df_all = atr(df_all,TICKER_LIST,14)

teim.sleep(60)

df_all = rvi(df_all,TICKER_LIST,_window=14)

df_all = sbgamma(df_all,TICKER_LIST)

time.sleep(30)

df_all = onhisma(df_all,TICKER_LIST,_window=5)
df_all = onlosma(df_all,TICKER_LIST,_window=5)

df_all = onhisma(df_all,TICKER_LIST,_window=21)
df_all = onlosma(df_all,TICKER_LIST,_window=21)

df_all = onhisma(df_all,TICKER_LIST,_window=34)
df_all = onlosma(df_all,TICKER_LIST,_window=34)

df_all = importohlc(df_all,weekly_all,TICKER_LIST,_suffix='_weekly')

df_all = importohlc(df_all=df_all,other_all=daily_all,TICKER_LIST=TICKER_LIST,_suffix='_daily')

engine.say("The job is done")
engine.runAndWait()

joblib.dump(df_all,'JOBLIB/Built_bases/df_all')
joblib.dump(daily_all,'JOBLIB/Built_bases/daily_all')
joblib.dump(weekly_all,'JOBLIB/Built_bases/weekly_all')

engine.say("AAll the bases are saved")
engine.runAndWait()

_t2 = dt.datetime.now()
print('Début des opérations horodatée à',col.Fore.YELLOW,dt.datetime.now(),col.Style.RESET_ALL)
print((_t2 - _t1))


if __name__ == "__main__":
    pass 