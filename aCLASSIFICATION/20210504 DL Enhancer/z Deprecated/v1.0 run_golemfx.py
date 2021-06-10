__author__ = 'LumberJack'
__copyright__ = 'D.A.G. 26 - 5781'

####################################################################
####################################################################
############################### GOLEM FX ###########################
####################################################################
####################################################################

from functools import reduce
from golemfx import *
import joblib
import pandas as pd
import colorama as col
import ffn
from matplotlib import pyplot as plt
import warnings
import talib

warnings.filterwarnings("ignore")

TIK = ['AUD','NZD','GBP','JPY','CHF','CAD','SEK','NOK','ILS','MXN','USD','EUR']
RATE = [0.776,0.721,1.3912,1/105.91,1/0.892,1/1.2681,1/8.2884,1/8.4261,1/3.2385,1/20.1564,1,1.21]
df_ratefx = pd.DataFrame(index=TIK)
df_ratefx['rate'] = RATE

def shuttle():
    return(x,_ticker)

x = 'EUR/USD'
_period = 'H1'
_period2 = 'H1'
_ticker = x.replace('/','')
_start = '2010-01-01' # start the train there '2010-01-01'
_mid = '2016-08-31' # stop the train and begin the test there '2016-08-31'
_stop = '2016-09-01' # stop the test there. After that, it is kept for oos
_last = '2021-04-09' # '2020-12-31'
_nb_bougie_exit = 5555555555
_trigger_reengage = 0
_trigger_target = 1
_trigger_invers = 0
_trigger_sl = 1
_verbose = 0
_cash_ini = 200000
_target = 0.004
_exposure = 2
_sl = 0.003
_rate = df_ratefx.loc[x[4:],'rate']
_size = _cash_ini / df_ratefx.loc[x[:3],'rate']
_trigger_spread = 0.025

joblib.dump(x,'VARS/'+_ticker+'/'+'x.dag')
joblib.dump(_ticker,'VARS/'+_ticker+'/'+'ticker.dag')
joblib.dump(_period ,'VARS/'+_ticker+'/'+'period.dag')
joblib.dump(_period2 ,'VARS/'+_ticker+'/'+'period2.dag')
joblib.dump(_start,'VARS/'+_ticker+'/'+'start.dag')
joblib.dump(_mid,'VARS/'+_ticker+'/'+'mid.dag')
joblib.dump(_stop ,'VARS/'+_ticker+'/'+'stop.dag')
joblib.dump(_last,'VARS/'+_ticker+'/'+'last.dag')
joblib.dump(_nb_bougie_exit ,'VARS/'+_ticker+'/'+'nbbougieexit.dag') 
joblib.dump(_trigger_reengage,'VARS/'+_ticker+'/'+'trig_reengage.dag')
joblib.dump(_trigger_target,'VARS/'+_ticker+'/'+'trig_target.dag')
joblib.dump(_trigger_invers ,'VARS/'+_ticker+'/'+'trig_invers.dag')
joblib.dump(_trigger_sl ,'VARS/'+_ticker+'/'+'trig_sl.dag')
joblib.dump(_verbose,'VARS/'+_ticker+'/'+'verbose.dag')
joblib.dump(_cash_ini,'VARS/'+_ticker+'/'+'cashini.dag')
joblib.dump(_target,'VARS/'+_ticker+'/'+'target.dag')
joblib.dump(_exposure,'VARS/'+_ticker+'/'+'exposure.dag')
joblib.dump(_sl,'VARS/'+_ticker+'/'+'sl.dag')
joblib.dump(df_ratefx,'VARS/'+_ticker+'/'+'df_ratefx.dag')
joblib.dump(_rate,'VARS/'+_ticker+'/'+'rate.dag')
joblib.dump(_size,'VARS/'+_ticker+'/'+'size.dag')
joblib.dump(_trigger_spread,'VARS/'+_ticker+'/'+'trig_spread.dag')

def DagMaxBase(df):
    """[Generate future_max for scaling]

    Args:
        df ([pandas]): [Generate a new]
    """ 
    # max([abs(df_train[label]).max(),abs(df_train[label]).max()])
       
    features_max = pd.DataFrame()
    try:
        for label in df.drop(['Date','Symbol','Signal'],axis=1).columns:
            features_max.loc[0,label] = abs(df[label]).max()
    except:
        for label in df.drop(['Date','Symbol'],axis=1).columns:
            features_max.loc[0,label] = abs(df[label]).max()   
    
    joblib.dump(features_max,x.replace('/','')+'_MAX.dag')
    return(features_max)


if __name__ == "__main__":

    print('__________________________________')
    print('     ___ Period 1 : => ', _period,' ___')
    print('     ___ Period 2 : => ', _period2,' ___')
    print('__________________________________')

    _token = joblib.load('TOKENS/_api_token.dag')
    _server = 'demo'

    try:
        con
    except NameError:
        con = fxcmpy.fxcmpy(access_token=_token, log_level='error',server=_server)
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
        engine.say("Gaulem is Connected")
        engine.runAndWait()
                
    else:
        con = conX(con,_token,_server)

    while True:
        print(col.Fore.GREEN,'\n\n')
        print('Entrez le mode désiré')
        print('1 : Initialisation de la base')
        print('2 : MAJ de la base existante')
        print('3 : Deep Learning')
        print('4 : Backtest')
        print('5 : Traitement Statistique')
        print('6 : Live')
        print('7 : Backtest sur Live')
        print(col.Style.RESET_ALL)

        _answer = input('Tapez la réponse : ')
        print('\n\n')

        # INITIALISATION PREMIERE DE LA BASE A APRTIR DE HDD
        if _answer == '1' :
            df,df_H1,df_D1,df_W1 = init_base()

        # MISE A JOUR DE LA BASE EXISTANTE  
        elif _answer == '2':
            _t10 = dt.datetime.now()
            print('\nMise à jour des données')
            print('Début des opérations horodatée à',col.Fore.YELLOW,dt.datetime.now(),col.Style.RESET_ALL)

            _decay = dt.datetime.now().hour - con.get_candles(x,period='m5',number=1).index.hour[0]
            joblib.dump(_decay,'VARS/'+_ticker+'/'+'decay.dag')

            df = joblib.load('BASES/'+_ticker+'_'+_period)
            df_H1 = joblib.load('BASES/'+_ticker+'_'+_period2)
            
            df = reduce_df(df)
            df_H1 = reduce_df(df_H1)
            
            df = maj(df,con,_period,_decay)
            df_H1 = maj(df_H1,con,_period2,_decay)

            df, df_H1, df_D1, df_W1 =  make_indicators(df, df_H1)

            print('Sauvegarde des Bases')
            joblib.dump(df_H1,'BASES/'+_ticker+'_'+_period2)
            joblib.dump(df,'BASES/'+_ticker+'_'+_period)
            joblib.dump(df_D1,'BASES/'+_ticker+'_D1')
            joblib.dump(df_W1,'BASES/'+_ticker+'_W1')

            print('Premier index de df: ',df.index[0])
            print('Dernier index de df: ',df.index[-1])
            print('shape de df',df.shape)
            print()
            print('Premier index de df_H1: ',df_H1.index[0])
            print('Dernier index de df_H1: ',df_H1.index[-1])
            print('shape de df_H1',df_H1.shape)
            print()

            _t20 = dt.datetime.now()
            print('\nFin des opérations horodatée à',col.Fore.YELLOW,dt.datetime.now(),col.Style.RESET_ALL)
            print('Executé en :',(_t20 - _t10),'\n')
            
        # DEEP LEARNING AVEC SAUVEGARDE DU MODELE
        elif _answer == '3' :
            df = joblib.load('BASES/'+_ticker+'_'+_period)
            
            df = strategy(df)

            print('shape1',df.shape)
            
            _year_bottom = _stop
            _year_top = _last
            print(col.Fore.RED,'###############################################################################################')
            print(' ####################################### OOS WITHOUT AI ########################################')
            print(' ###############################################################################################',col.Style.RESET_ALL)
            ##### Backtest Over Night
            TRACKER,_nb_looser = bt(df,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                    _rate,x,_target,_exposure,_size,_sl)

            _year_bottom = _start
            _year_top = _stop
            print()
            print(col.Fore.CYAN,'###############################################################################################')
            print(' #################################### TRAIN/TEST WITHOUT AI ####################################')
            print(' ###############################################################################################',col.Style.RESET_ALL)


            TRACKER,_nb_looser = bt(df,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                    _rate,x,_target,_exposure,_size,_sl)

            print()
            print(col.Fore.BLUE,'###############################################################################################')
            print(' #################################### DENOISING & ENHANCING ####################################')
            print(' ###############################################################################################',col.Style.RESET_ALL)
            while _nb_looser > 0 :
                
                df['TRACKER'] = np.where(df.index.isin(TRACKER),1,0)
                df['Valid'] = np.where(((df.Signal!=0)&(df.TRACKER==1)),1,0)
                df['Signal'] = np.where(((df.Valid==1)&(df.Signal==1)),1,np.where(((df.Valid==1)&(df.Signal==-1)),-1,0))
                ##### Purification of signal by denoising and enhancing
                TRACKER,_nb_looser = bt(df,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                        _rate,x,_target,_exposure,_size,_sl)
            
            df = strategy(df)

            print('shape1',df.shape)
            
            features = featuring(df)
            print('Shape de features',features.shape)
            features['TRACKER'] = np.where(features.index.isin(TRACKER),1,0)

            # First, we must have an output. We'll call it 'Valid'. It wil be where Tracker & Signal are both to 1
            features['Valid'] = np.where(((features.Signal!=0)&(features.TRACKER==1)),1,0) # Don't miss the point that even a Signal -1 must be considered as a good one by TRACKER

            # And drop the nan
            features = features.dropna()
            ##### Signal is from strategy. This is potential good one. But we have to create the TRACKER column where the Signal where efficient

            features_train, features_test, features_oos = split_learn(features)

            print('Shape de features',features.shape,features_test.shape,features_oos.shape)

            features_max = joblib.load('VARS/'+_ticker+'/'+_ticker+'_MAX.dag')

            for label in features_train.drop(['Date','Symbol','Signal','TRACKER','Valid'],axis=1).columns:
                features_train[label] = features_train[label] / features_max[label][0]
            for label in features_test.drop(['Date','Symbol','Signal','TRACKER','Valid'],axis=1).columns:
                features_test[label] = features_test[label] / features_max[label][0]
            for label in features_oos.drop(['Date','Symbol','Signal','TRACKER','Valid'],axis=1).columns:
                features_oos[label] = features_oos[label] / features_max[label][0]

            '''features_train = scaling(features_train)
            features_test = scaling(features_test)
            features_oos = scaling(features_oos)'''

            _model = learning(features_train, features_test, _save=1)
            
            
            df_oos = df[(df.Date > _stop)&(df.Date <= _last)].dropna()

            df_oos = df_oos[df_oos.Symbol==x.replace('/','')]


            df_oos['Valid'] = _model.predict(features_oos.drop(['Date','Symbol','Signal','TRACKER','Valid'],axis=1))

            #df_all_oos = df_all_oos.dropna()

            df_oos['Signal'] = np.where((df_oos.Signal==1)&(df_oos.Valid==1),1,np.where((df_oos.Signal==-1)&(df_oos.Valid==1),-1,0))
            
            ##### On backtest selon le ticker selectionné sur la période déterminée

            print(x)
            _year_bottom = _stop
            _year_top = _last
            


            ##### Backtest Over Night
            FINAL_TRACKER = bt(df_oos,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                    _rate,x,_target,_exposure,_size,_sl)
            

        # BACKTEST DE LA BASE A JOUR
        elif _answer == '4' :

            _year_bottom = _stop
            _year_top = _last

            df = joblib.load('BASES/'+_ticker+'_'+_period)

            df = df[(df.index >= _mid)&(df.index <=_last)]
            
            df = strategy(df)

            features = featuring(df)
            print('Shape de features, df',features.shape, df.shape)

            # And drop the nan
            features = features.dropna()
            ##### Signal is from strategy. This is potential good one. But we have to create the TRACKER column where the Signal where efficient

            df = df[(df.index >= features.index[0])&(df.index <=features.index[-1])]

            print('Shape de features, df, après features.dropna(), ',features.shape, df.shape)
            print()
            print('features.ini et df.ini', features.index[0],df.index[0])
            print('features.fini et df.fini', features.index[-1],df.index[-1])
            print()

            features_max = joblib.load('VARS/'+_ticker+'/'+_ticker+'_MAX.dag')

            for label in features.drop(['Date','Symbol','Signal'],axis=1).columns:
                features[label] = features[label] / features_max[label][0]
           
            _model = joblib.load('MODELS/Save_'+_ticker+'_'+_period+'.dag')
            
            df['Valid'] = _model.predict(features.drop(['Date','Symbol','Signal'],axis=1))

            #df_all_oos = df_all_oos.dropna()

            df['Signal'] = np.where((df.Signal==1)&(df.Valid==1),1,np.where((df.Signal==-1)&(df.Valid==1),-1,0))
            
            ##### On backtest selon le ticker selectionné sur la période déterminée

            print(x)
          
            ##### Backtest Over Night
            
            SUBFINAL_TRACKER = bt(df,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                    _rate,x,_target,_exposure,_size,_sl)
             
        # TRAITEMENTS STATISTIQUES
        elif _answer == '5' :
            print(col.Fore.CYAN+'\n\nTRAITEMENT STATISTIQUE DES DONNEES\n',col.Style.RESET_ALL)
            candle_feedback = joblib.load('BT/'+_ticker+'_candle_feedback.dag')
            candle_feedback = timerange1D(candle_feedback)
            stats = candle_feedback.drop(['Symbol','Size','Date'],axis=1).calc_stats()
            stats[_ticker].display()
            print(col.Fore.CYAN+'\n\nMONTHLY RETURN FOR '+x+'\n',col.Style.RESET_ALL)
            stats[_ticker].display_monthly_returns()
        
        # LIVE
        elif _answer == '6' : 
            go_live(con)

        # BACKTEST SUR LE LIVE
        elif _answer == '7' : 
            _year_bottom = input('Entrez la date de début : ')
            _year_top = input('Entrez la date de fin : ')
            _verbose = 1
            
            df = joblib.load('BASES/'+_ticker+'_'+_period)

            df = df[(df.index >= _mid)&(df.index <=_last)]
            
            df = strategy(df)

            features = featuring(df)
            print('Shape de features, df',features.shape, df.shape)

            # And drop the nan
            features = features.dropna()
            ##### Signal is from strategy. This is potential good one. But we have to create the TRACKER column where the Signal where efficient

            df = df[(df.index >= features.index[0])&(df.index <=features.index[-1])]

            print('Shape de features, df, après features.dropna(), ',features.shape, df.shape)
            print()
            print('features.ini et df.ini', features.index[0],df.index[0])
            print('features.fini et df.fini', features.index[-1],df.index[-1])
            print()

            features_max = joblib.load('VARS/'+_ticker+'/'+_ticker+'_MAX.dag')

            for label in features.drop(['Date','Symbol','Signal'],axis=1).columns:
                features[label] = features[label] / features_max[label][0]
           
            _model = joblib.load('MODELS/Save_'+_ticker+'_'+_period+'.dag')
            
            df['Valid'] = _model.predict(features.drop(['Date','Symbol','Signal'],axis=1))

            #df_all_oos = df_all_oos.dropna()

            df['Signal'] = np.where((df.Signal==1)&(df.Valid==1),1,np.where((df.Signal==-1)&(df.Valid==1),-1,0))
            
            ##### On backtest selon le ticker selectionné sur la période déterminée

            print(x)
          
            ##### Backtest Over Night
            
            SUBFINAL_TRACKER = bt(df,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                    _rate,x,_target,_exposure,_size,_sl)


        else :
            print('Mauvaise réponse. Essayez à nouveau.')
        