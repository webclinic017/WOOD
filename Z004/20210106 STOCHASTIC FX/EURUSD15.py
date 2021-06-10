
if __name__ == "__main__":
        
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

        print('version fxcmpy :',fxcmpy.__version__)


        #if __name__ == "__main__":
            #if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            #   os.chdir(sys._MEIPASS)
        _token = 'dbdc379ce7761772c662c3e92250a0ae38385b2c'
        _server = 'demo'
        _user_id = 'D261282181'
        _compte = '01215060'
        _password = 'waXz1'

        _period = 'm15'
        _name = 'MLPClassifier'

        TICKER_LIST = ['EUR/USD']
        x = TICKER_LIST[0]
        TIK = ['AUD','NZD','GBP','JPY','CHF','CAD','SEK','NOK','ILS','MXN','USD','EUR']
        RATE = [0.776,0.721,1.3912,1/105.91,1/0.892,1/1.2681,1/8.2884,1/8.4261,1/3.2385,1/20.1564,1,1]
        df_ratefx = pd.DataFrame(index=TIK)
        df_ratefx['rate'] = RATE
        _scaler = MaxAbsScaler()
        savename = 'Save_'+x.replace('/','')+'_m15.sav'
        _model = joblib.load(savename)
        df_all = joblib.load('EURUSD_'+_period)
        df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
        hourly_all = joblib.load('EURUSD_H1')
        hourly_all = hourly_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]

        def get_all_data(TICKER_LIST,_period):
            # print(col.Fore.YELLOW+'\nRécupération des data intraday'+col.Style.RESET_ALL)
            _ticker = TICKER_LIST[0]
            _ticker = _ticker.replace('/','')
            
            # print(col.Fore.BLUE,'Ticker',col.Fore.YELLOW,_ticker[:3]+'/'+_ticker[3:],col.Style.RESET_ALL)
            ##### Chargement de la base par ticker
            df_all = pd.read_csv('../BASES/Base/'+_ticker+'_'+_period+'_BidAndAsk.csv')

            ##### Ajout de la colonne Symbol pour identifier le ticker
            df_all['Symbol'] = _ticker

            ##### On fixe la date en index sous forme de Timestamp
            df_all['Lindex'] = pd.to_datetime(df_all['Date'] + ' ' + df_all['Time'])
            df_all.set_index(pd.to_datetime(df_all.Lindex,format='%Y-%m-%d %H:%M:%S'),drop=True,inplace=True)

            ###### On drop les colonnes inutiles
            df_all = df_all.drop(['Date','Lindex','Time','Total Ticks'],axis=1)

            ##### On enlève les jours correspondant au samedi et au dimanche
            df_all['WE'] = np.where(((df_all.index.weekday == 5) | (df_all.index.weekday == 6)),None,df_all.index.weekday)
            df_all = df_all.dropna()
            df_all = df_all.drop(['WE'],axis=1)

            ##### Calcul des averages pour les OHLC
            df_all['Open'] = (df_all['OpenBid'] + df_all['OpenAsk']) / 2
            df_all['High'] = (df_all['HighBid'] + df_all['HighAsk']) / 2
            df_all['Low'] = (df_all['LowBid'] + df_all['LowAsk']) / 2
            df_all['Close'] = (df_all['CloseBid'] + df_all['CloseAsk']) / 2

            return(df_all.sort_index(axis=1))

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

        def smaratio(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul SMA'+col.Style.RESET_ALL)
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
            return(df_all.sort_index(axis=0))

        def sma(df_all,_window=200,_plot=0,_ticker=None,start=None,end=None):
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

        def atrratio(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul ATR RATIO'+col.Style.RESET_ALL)
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
                # print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,i,col.Style.RESET_ALL,end='',flush=True)
                TR_data = df_all[df_all.Symbol == i].copy()
                df_all.loc[df_all.Symbol==i,'ATR_'+str(_fast)] = Wilder(TR_data['TR'], _fast)
                df_all.loc[df_all.Symbol==i,'ATR_'+str(_slow)] = Wilder(TR_data['TR'], _slow)

            df_all['ATR_Ratio'] = df_all['ATR_'+str(_fast)] / df_all['ATR_'+str(_slow)]
            
            df_all = df_all.drop(['prev_close','TR'],axis=1)
            return(df_all.sort_index(axis=0))

        def adx(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul ADX'+col.Style.RESET_ALL)
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
                # print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,i,col.Style.RESET_ALL,end='',flush=True)
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
                # print('\r',col.Fore.BLUE,'Ticker',col.Fore.YELLOW,i,col.Style.RESET_ALL,end='',flush=True)
                ADX_data = df_all[df_all.Symbol == i].copy()
                df_all.loc[df_all.Symbol==i,'ADX_'+str(_fast)] = Wilder(ADX_data['DX_'+str(_fast)], _fast)
                df_all.loc[df_all.Symbol==i,'ADX_'+str(_slow)] = Wilder(ADX_data['DX_'+str(_slow)], _slow)

            df_all = df_all.drop(['DX_'+str(_fast),'DX_'+str(_slow),'+DI_'+str(_fast),'-DI_'+str(_fast),'+DI_'+str(_slow),'-DI_'+str(_slow),'-DM','+DM','prev_high','prev_low'],axis=1)
            return(df_all.sort_index(axis=0))

        def slowstochastic(df_all,TICKER_LIST,_window=5,_per=3,_plot=0,_ticker=None,start=None,end=None):
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
            

        def faststochastic(df_all,TICKER_LIST,_window=5,_per=3,_plot=0,_ticker=None,start=None,end=None):
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
            df_1['fast_K'+str(_window)] = (((df_1['Close'] - df_1['Lowest_'+str(_window)])/(df_1['Highest_'+str(_window)] - df_1['Lowest_'+str(_window)]))*100)

            ##### On smoothering le stochastic en calculant la moyenne sur les fenetres slow et fast de ces valeurs
            df_1['fast_D'+str(_window)] = df_1['slow_K'+str(_window)].rolling(window = _per).mean()
            df_1 = df_1.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)
            return(df_1.sort_index(axis=0))
            

        def fullstochastic(df_all,TICKER_LIST,_window=5,_per1=3,_per2=3,_plot=0,_ticker=None,start=None,end=None):
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
            df_1['full_K'+str(_window)] = (((df_1['Close'] - df_1['Lowest_'+str(_window)])/(df_1['Highest_'+str(_window)] - df_1['Lowest_'+str(_window)]))*100).rolling(window = _per1).mean()

            ##### On smoothering le stochastic en calculant la moyenne sur les fenetres slow et fast de ces valeurs
            df_1['full_D'+str(_window)] = df_1['slow_K'+str(_window)].rolling(window = _per2).mean()

            df_1 = df_1.drop(['Lowest_'+str(_window),'Highest_'+str(_window)],axis=1)

            return(df_1.sort_index(axis=0))
            
            

        def rsiratio(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul RSI'+col.Style.RESET_ALL)
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

            return(df_all.sort_index(axis=0))

        def rsi(df_all,_window=5,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul RSI'+col.Style.RESET_ALL)
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

            return(df_all.sort_index(axis=0))

        def macd(df_all,_fast=5,_slow=15,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul MACD'+col.Style.RESET_ALL)
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

            return(df_all.sort_index(axis=0))

        def bollinger(df_all,_slow=15,_plot=0,_ticker=None,start=None,end=None):
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

        def rc(df_all,_slow=15,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul RATE OF CHANGE'+col.Style.RESET_ALL)
            '''Rate of Change
            Rate of change is a momentum indicator that explains a price momentum relative to a price fixed period before.
            
            df_all = La base à travailler, _fast = fenetre courte, _slow = fenetre longue,
            _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plo'''

            df_all['RC'] = df_all.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods = _slow)) 

            return(df_all.sort_index(axis=0))


        def onlosma(df_all,TICKER_LIST,_window=8,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul ONLOSMA'+col.Style.RESET_ALL)

            '''df_all = La base à travailler, _fast = fenetre courte,
            _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''

            # print('On High Simple Moving Average Calculation')
            _ticker = TICKER_LIST[0]
            _ticker = _ticker.replace('/','')
            hourly = df_all[df_all.Symbol==_ticker].copy()
            hourly['ONLOSMA_'+str(_window)] = hourly.Low.rolling(_window).mean()
            return(hourly.sort_index(axis=0))
            
        def onhisma(df_all,TICKER_LIST,_window=8,_plot=0,_ticker=None,start=None,end=None):
            # print(col.Fore.MAGENTA+'\nCalcul ONHISMA'+col.Style.RESET_ALL)
            '''df_all = La base à travailler, _fast = fenetre courte,
            _plot=0 par defaut et 1 si plot, _ticker=None ou si _plot=1 le ticker à ploter,start=debut du plot, end=fin du plot'''
            _ticker = TICKER_LIST[0]
            _ticker = _ticker.replace('/','')
            hourly = df_all[df_all.Symbol==_ticker].copy()
            hourly['ONHISMA_'+str(_window)] = hourly.High.rolling(_window).mean()
            return(hourly.sort_index(axis=0))

        def atr(df_all,TICKER_LIST,_window=14,_plot=0,_ticker=None,start=None,end=None):
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

        def pivotimportdaily(daily_all,weekly_all,TICKER_LIST):
            # print(col.Fore.MAGENTA+'\nCalcul du PIVOT IMPORT'+col.Style.RESET_ALL)
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

            _ticker = TICKER_LIST[0]
            _ticker = _ticker.replace('/','')
            for i in range(len(daily_all)):
                PP.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].PP[0])
                S38.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].S38[0])
                S62.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].S62[0])
                S100.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].S100[0])
                S138.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].S138[0])
                S162.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].S162[0])
                S200.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].S200[0])
                R38.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].R38[0])
                R62.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].R62[0])
                R100.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].R100[0])
                R138.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].R138[0])
                R162.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].R162[0])
                R200.append(weekly_all[(weekly_all.Date.dt.isocalendar().week == daily_all.Date.dt.isocalendar().week[i])&\
                        (weekly_all.Date.dt.year == daily_all.Date.dt.year[i])].R200[0])
            daily_all['PP'] = PP
            daily_all['S38'] = S38
            daily_all['S62'] = S62
            daily_all['S100'] = S100
            daily_all['S138'] = S138
            daily_all['S162'] = S162
            daily_all['S200'] = S200
            daily_all['R38'] = R38
            daily_all['R62'] = R62
            daily_all['R100'] = R100
            daily_all['R138'] = R138
            daily_all['R162'] = R162
            daily_all['R200'] = R200
            return(daily_all.sort_index(axis=0))

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
            return(hourly.sort_index(axis=0))
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
            """[Entrer la df préparée avec les bons indictaurs. Renvoie une nouvelle df qu'avec les features + Symbol + Date + Signal]

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

        def polytrans(features,features_test,features_oos,poly):
            """[Transformation par les Quantile]

            Args:
                features ([dataframe]): [train]
                features_test ([dataframe]): [test]
                features_oos ([dataframe]): [oos]
                poly.fit_transform ([sklearn]): [from preprocessing] 
            """    
            
            features['FEMA_21'] = poly.fit_transform(np.nan_to_num(features.FEMA_21.astype(np.float32)).reshape(-1, 1))
            features['FEMA_8'] = poly.fit_transform(np.nan_to_num(features.FEMA_8.astype(np.float32)).reshape(-1, 1))
            features['FADRLo'] = poly.fit_transform(np.nan_to_num(features.FADRLo.astype(np.float32)).reshape(-1, 1))
            features['FADRHi'] = poly.fit_transform(np.nan_to_num(features.FADRHi.astype(np.float32)).reshape(-1, 1))
            features['FRVI40'] = poly.fit_transform(np.nan_to_num(features.FRVI40.astype(np.float32)).reshape(-1, 1))
            features['FRVI60'] = poly.fit_transform(np.nan_to_num(features.FRVI60.astype(np.float32)).reshape(-1, 1))
            features['FONLOSMA5'] = poly.fit_transform(np.nan_to_num(features.FONLOSMA5.astype(np.float32)).reshape(-1, 1))
            features['FONHISMA5'] = poly.fit_transform(np.nan_to_num(features.FONHISMA5.astype(np.float32)).reshape(-1, 1))
            features['FONLOSMA21'] = poly.fit_transform(np.nan_to_num(features.FONLOSMA21.astype(np.float32)).reshape(-1, 1))
            features['FONHISMA21'] = poly.fit_transform(np.nan_to_num(features.FONHISMA21.astype(np.float32)).reshape(-1, 1))
            features['FONLOSMA34'] = poly.fit_transform(np.nan_to_num(features.FONLOSMA34.astype(np.float32)).reshape(-1, 1))
            features['FSBGAMMA'] = poly.fit_transform(np.nan_to_num(features.FSBGAMMA.astype(np.float32)).reshape(-1, 1))
            features['FOPENWEEKLY'] = poly.fit_transform(np.nan_to_num(features.FOPENWEEKLY.astype(np.float32)).reshape(-1, 1))
            features['FHIGHWEEKLY'] = poly.fit_transform(np.nan_to_num(features.FHIGHWEEKLY.astype(np.float32)).reshape(-1, 1))
            features['FLOWWEEKLY'] = poly.fit_transform(np.nan_to_num(features.FLOWWEEKLY.astype(np.float32)).reshape(-1, 1))
            features['FCLOSEWEEKLY'] = poly.fit_transform(np.nan_to_num(features.FCLOSEWEEKLY.astype(np.float32)).reshape(-1, 1))
            features['FOPENDAILY'] = poly.fit_transform(np.nan_to_num(features.FOPENDAILY.astype(np.float32)).reshape(-1, 1))
            features['FHIGHDAILY'] = poly.fit_transform(np.nan_to_num(features.FHIGHDAILY.astype(np.float32)).reshape(-1, 1))
            features['FLOWDAILY'] = poly.fit_transform(np.nan_to_num(features.FLOWDAILY.astype(np.float32)).reshape(-1, 1))
            features['FCLOSEDAILY'] = poly.fit_transform(np.nan_to_num(features.FCLOSEDAILY.astype(np.float32)).reshape(-1, 1))
            features['FOPENHOURLY'] = poly.fit_transform(np.nan_to_num(features.FOPENHOURLY.astype(np.float32)).reshape(-1, 1))
            features['FHIGHHOURLY'] = poly.fit_transform(np.nan_to_num(features.FHIGHHOURLY.astype(np.float32)).reshape(-1, 1))
            features['FLOWHOURLY'] = poly.fit_transform(np.nan_to_num(features.FLOWHOURLY.astype(np.float32)).reshape(-1, 1))
            features['FCLOSEHOURLY'] = poly.fit_transform(np.nan_to_num(features.FCLOSEHOURLY.astype(np.float32)).reshape(-1, 1))
            features['FSMA200'] = poly.fit_transform(np.nan_to_num(features.FSMA200.astype(np.float32)).reshape(-1, 1))
            features['FBOLUP20'] = poly.fit_transform(np.nan_to_num(features.FBOLUP20.astype(np.float32)).reshape(-1, 1))
            features['FPP'] = poly.fit_transform(np.nan_to_num(features.FPP.astype(np.float32)).reshape(-1, 1))
            features['FS38'] = poly.fit_transform(np.nan_to_num(features.FS38.astype(np.float32)).reshape(-1, 1))
            features['FS62'] = poly.fit_transform(np.nan_to_num(features.FS62.astype(np.float32)).reshape(-1, 1))
            features['FS100'] = poly.fit_transform(np.nan_to_num(features.FS100.astype(np.float32)).reshape(-1, 1))
            features['FS138'] = poly.fit_transform(np.nan_to_num(features.FS138.astype(np.float32)).reshape(-1, 1))
            features['FR162'] = poly.fit_transform(np.nan_to_num(features.FS162.astype(np.float32)).reshape(-1, 1))
            features['FS200'] = poly.fit_transform(np.nan_to_num(features.FS200.astype(np.float32)).reshape(-1, 1))
            features['FR38'] = poly.fit_transform(np.nan_to_num(features.FR38.astype(np.float32)).reshape(-1, 1))
            features['FR62'] = poly.fit_transform(np.nan_to_num(features.FR62.astype(np.float32)).reshape(-1, 1))
            features['FR100'] = poly.fit_transform(np.nan_to_num(features.FR100.astype(np.float32)).reshape(-1, 1))
            features['FR138'] = poly.fit_transform(np.nan_to_num(features.FR138.astype(np.float32)).reshape(-1, 1))
            features['FR162'] = poly.fit_transform(np.nan_to_num(features.FR162.astype(np.float32)).reshape(-1, 1))
            features['FR200'] = poly.fit_transform(np.nan_to_num(features.FR200.astype(np.float32)).reshape(-1, 1))
            features['SBATR'] = poly.fit_transform(np.nan_to_num(features.SBATR.astype(np.float32)).reshape(-1, 1))
            
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

            return(features,features_test,features_oos)

        def bt(price,_year_bottom,_year_top,_nb_bougie_exit,_trigger_reengage,_trigger_target,_trigger_invers,_trigger_sl,_verbose,_cash_ini,\
                _rate,x,_target,_exposure,_size,_sl):
            engine = pyttsx3.init()

            print('Librairies imported\n')

            engine.say("Backtesting in progress")
            engine.runAndWait()

            #from numpy import loadtxt
            #from functools import reduce
            _t1 = dt.datetime.now()
            print('Début des opérations horodatée à',dt.datetime.now())

            _total = 0
            _cash = _cash_ini
            _pnl = 0

            #_flag = 0

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
            _cash = _cash_ini
            _tracker = 0
            WINNERS = []
            LOOSERS = []
            SIGNAL = []
            OPEN = []
            OPEN_BID = []
            OPEN_ASK = []
            EXPO_MAX = []
            TRACKER = []
            LONGWINNERS = []
            LONGLOOSERS = []
            SHORTWINNERS = []
            SHORTLOOSERS = []


            df_resultats = pd.DataFrame(index=['Equity','Nbre Winners','Nbre winners long','Nbre winners short','Nbre Loosers','Nbre loosers long','Nbre loosers short','Max lenght of trade','Min lenght of trade',\
                'Average lenght of trade','Cumul pnl'])
            
            print('\nChargement de la nouvelle base\n\n')

            engine.say("קדימה")
            engine.runAndWait()

            print(col.Fore.MAGENTA,'Le rate du ticker',x,'est à ',_rate,col.Style.RESET_ALL)

            price = price[(price.index >= _year_bottom) & (price.index <= _year_top)]
            print('Bases chargées')

            print('TETEL process effectué')

            print(col.Fore.CYAN,'ENTERING THE BACKTEST',col.Style.RESET_ALL)

            time.sleep(0.2)
                
            price = price.dropna()

            _position = 0
            _equity = 0
            _nbtransactions = 0
            backtest_graph = pd.DataFrame()
            EQUITY = [_cash]
            CASH = [_cash]
            _winner = 0
            _looser = 0
            _longwinner = 0
            _longlooser = 0
            _shortwinner = 0
            _shortlooser = 0
            _index_entry = 0
            TRADE_DURATION = []
            _average_duration = 0


            PRICE_BUY = []
            PRICE_SELL = []

            _total = 0

            

            _open_buy = 0
            _open_sell = 0

            for i in tqdm(range(0,len(price))):

                ##### POSITIONS EN L'AIR 
                if i >= (len(price)-1) and (_position == 1 or _position == -1) :

                    if _position == -1:
                        _position = 99
                        _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell
                        _total += _pnl
                        _cash += _pnl
                        _equity = _cash
                        EQUITY.append(_equity)
                        CASH.append(_cash)
                        
                        if _pnl > 0:
                            _winner += _open_sell
                            _longwinner+=_open_sell
                        else:
                            _looser += _open_sell
                            _shortlooser +=_open_sell
                        
                        TRADE_DURATION.append(i - _index_entry)
                        
                        if _verbose == 1:
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
                        _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy
                        _total += _pnl
                        _cash += _pnl
                        _equity = _cash
                        EQUITY.append(_equity)
                        CASH.append(_cash)
                        
                        if _pnl > 0:
                            _winner += _open_buy
                            _longwinner +=_open_buy
                        else:
                            _looser += _open_buy
                            _longlooser += _open_buy

                        TRADE_DURATION.append(i - _index_entry)
                        if _verbose == 1:
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
                        if _verbose == 1:
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
                        if _verbose == 1:
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
                        continue
                
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
                        if _verbose == 1:
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

                    ### CLOSE LONG ON INVERSE SIGNAL
                    if price.Signal[i] == -1 and _trigger_invers == 1:
                        _position = 0
                        _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy
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
                        if _verbose == 1:
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
                        _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy
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
                        if _verbose == 1:
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
                    if (float(price.CloseBid.iloc[i]) - float(_price_buy_mean))/float(_price_buy_mean) >= _target and _trigger_target == 1:
                        _position = 0
                        _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy
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
                        if _verbose == 1:
                            print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                            if _pnl < 0:
                                print(_open_buy,'positions (TG) closed at',price.CloseBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                            else :
                                print(_open_buy,'positions (TG) closed at',price.CloseBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
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

                    # CLOSE LONG ON STOP LOSS
                    if (float(price. CloseBid.iloc[i]) - float(_price_buy_mean))/float(_price_buy_mean) <= - _sl and _trigger_sl == 1:
                        _position = 0
                        _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy
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
                        if _verbose == 1:
                            print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                            if _pnl < 0:
                                print(_open_buy,'positions (SL) closed at',price.CloseBid.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                            else :
                                print(_open_buy,'positions (SL) closed at',price.CloseBid.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
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
                    
                    else:

                        _pnl = (price.CloseBid.iloc[i] - _price_buy_mean) * _size * _open_buy
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
                        if _verbose == 1:
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

                    ### CLOSE SHORT ON INVERSE SIGNAL
                    if price.Signal[i] == 1 and _trigger_invers == 1:   
                        _position = 0
                        _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell
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
                        if _verbose == 1:
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
                        _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell
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
                        if _verbose == 1:
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
                    if (float(price.CloseAsk.iloc[i]) - float(_price_sell_mean))/float(_price_sell_mean) <= -_target and _trigger_target == 1:
                        _position = 0
                        _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell
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
                        if _verbose == 1:
                            print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                            if _pnl < 0 :    
                                print(_open_sell,'position (TG) closed at',price.CloseAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                            else:
                                print(_open_sell,'position (TG) closed at',price.CloseAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
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

                    ### CLOSE SHORT ON STOP LOSS
                    if (float(price.CloseAsk.iloc[i]) - float(_price_sell_mean))/float(_price_sell_mean) > _sl and _trigger_sl == 1:
                        _position = 0
                        _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell
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
                        if _verbose == 1:
                            print(col.Fore.MAGENTA,price.index[i].strftime(format='%Y-%m-%d %H:%M:%S'),col.Style.RESET_ALL)
                            if _pnl < 0 :    
                                print(_open_sell,'position (SL) closed at',price.CloseAsk.iloc[i],col.Fore.RED,'pnl', _pnl,col.Style.RESET_ALL)
                            else:
                                print(_open_sell,'position (SL) closed at',price.CloseAsk.iloc[i],col.Fore.GREEN,'pnl', _pnl,col.Style.RESET_ALL)
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

                    else:

                        _pnl = - (price.CloseAsk.iloc[i] - _price_sell_mean) * _size * _open_sell
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
                print("(No Duration)") 
                _average_duration = 'NA'
                _max_duration = 0.00002
                _min_duration = 0.00001 
            print(col.Fore.BLUE,'For ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL)
            if _total > 0:              
                print(col.Fore.MAGENTA,"\nTotal Profit & Loss : $",col.Fore.GREEN,round(_total * _rate,2),'. En ',\
                    _nbtransactions,col.Style.RESET_ALL,' transactions.' )
            else:
                print(col.Fore.MAGENTA,"\nTotal Profit & Loss : $",col.Fore.RED,round(_total * _rate,2),'. En ',\
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
            _generated_cash = round(df_historical.Pnl_Real.sum() * _rate,2)
            _generated_cash_perc = round((_generated_cash / _cash_ini) * 100,2)
            print(col.Fore.YELLOW,x,col.Fore.BLUE,'results',col.Style.RESET_ALL)
            print(col.Fore.MAGENTA,'Tested Period',_year_bottom,' à',_year_top,col.Style.RESET_ALL)
            print(col.Fore.CYAN,'Total Number of trades',max([sum(TOTAL_OPEN),sum(TOTAL_CLOSE)]),col.Style.RESET_ALL)
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
            print('Cumulated gains',round(df_historical[df_historical.Pnl_Real>0].Pnl_Real.sum() * _rate,2))
            print('Cumulated losses',round(df_historical[df_historical.Pnl_Real<0].Pnl_Real.sum() * _rate,2))
            print(col.Fore.BLUE,'PROFIT FACTOR : ',\
                abs(round(df_historical[df_historical.Pnl_Real>0].Pnl_Real.sum()/df_historical[df_historical.Pnl_Real<0].Pnl_Real.sum(),2)),col.Style.RESET_ALL)
            try:
                print(col.Fore.CYAN,'Winners Ratio :',\
                    round((df_resultats.T['Nbre Winners'].sum()*100)/(df_resultats.T['Nbre Loosers'].sum()+df_resultats.T['Nbre Winners'].sum()),2),\
                        '%',col.Style.RESET_ALL)
            except:
                print(col.Fore.CYAN,'Winners Ratio  :None',col.Style.RESET_ALL)
            try:
                print('Average Winners',round(sum(list(filter(lambda x:  x > 0,PNL_REAL)))/len(list(filter(lambda x:  x > 0,PNL_REAL))) * _rate,2))
                print('% Average Winners',round(sum(list(filter(lambda x:  x > 0,PNL_REAL)))/len(list(filter(lambda x:  x > 0,PNL_REAL))) * _rate * 100 / _cash_ini,2))
            except:
                print('No winner')
            try:
                print('Average Loosers',round(sum(list(filter(lambda x:  x < 0,PNL_REAL)))/len(list(filter(lambda x:  x < 0,PNL_REAL))) * _rate,2))
                print('% Average Loosers',round(sum(list(filter(lambda x:  x < 0,PNL_REAL)))/len(list(filter(lambda x:  x < 0,PNL_REAL))) * _rate / _cash_ini * 100,2))
            except:
                print('No looser')
            try:
                print('Average pnl',round(sum(PNL_REAL)/sum(TOTAL_OPEN) * _rate,2))
                print('% Average pnl',round((sum(PNL_REAL)/len(set(PNL_REAL)) * _rate) / _cash_ini * 100,2))
            except:
                print('No trade')
            
            print('Number of opened trades',sum(TOTAL_OPEN))
            print('Number of closed trades',sum(TOTAL_CLOSE))
            print('Max Exposure',max(EXPO_MAX),'x ',_size,'= ',max(EXPO_MAX)*_size,'$')
            return(TRACKER,df_resultats.T['Nbre Loosers'].sum())

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

        def buy(_period):
            print(dt.datetime.now())
            ##### BUY 
            _price = round(con.get_candles(x,period=_period,number=1).askclose[-1],5)
            _time = con.get_candles(x,period=_period,number=1).index[-1]
            _amount = 200
            _limit = round(_price + _price * 0.004,5)
            _stop = round(_price  - _price * 0.002,5)
            _atmarket = 3
            order = con.open_trade(symbol=x,is_buy=True, is_in_pips=False, amount=_amount, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)
            print(" Bougie de l'opération d'éxecution",col.Fore.BLUE,_time,col.Style.RESET_ALL)
            print(col.Fore.GREEN,'Achat sur le ticker',col.Fore.YELLOW,x,col.Fore.GREEN,'demandé à ',col.Fore.CYAN,_price,col.Style.RESET_ALL)

        def sell(_period):
            print(dt.datetime.now())
            _atmarket = 3
            _price = round(con.get_candles(x,period=_period,number=1).bidclose[-1],5)
            _time = con.get_candles(x,period=_period,number=1).index[-1]
            _amount = 200
            _stop = round(_price + _price * 0.002,5)
            _limit = round(_price  - _price * 0.004,5)
            order = con.open_trade(symbol=x,is_buy=False, is_in_pips=False, amount=_amount, time_in_force='IOC',order_type='MarketRange',limit=_limit,stop=_stop, at_market=3)
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
            df_all = df_all.append(addon.iloc[1:-1,:])
            df_all['WE'] = np.where(((df_all.index.weekday == 5) | (df_all.index.weekday == 6)),None,df_all.index.weekday)
            df_all = df_all.dropna()
            df_all =df_all.drop(['WE'],axis=1)
            joblib.dump(df_all,_path)
            return(df_all)

            ___Author___='LumberJack Jyss'
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

        #df_all = joblib.load(_path)
        #df_all = df_all[df_all.Symbol==x.replace('/','')]
        #df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
        #engine.say("Database is loaded. Ready to enter Live")
        #engine.runAndWait()
        print('\rBase Chargée.',end='',flush=True)
        while True:

            engine.say("Building the base")
            engine.runAndWait()
            print('\nConstruction de la base...')
            ##########
            
            print(x)
            
            print('\nWaiting for the candle...')
            print()

            ##########

            while dt.datetime.now().minute not in [0,15,30,45]:
                print('\rTicker tracké :',x,' ',dt.datetime.now(),end='',flush=True)
                time.sleep(1)
            print()
            while con.get_candles(x,period=_period,start=dt.datetime(df_all.index[-1].year,df_all.index[-1].month,df_all.index[-1].day,df_all.index[-1].hour,df_all.index[-1].minute)\
                ,end=dt.datetime.now()).index[-1].minute != dt.datetime.now().minute:

                time.sleep(0.5)
                
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
            df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]
            df_all = df_all.append(addon.iloc[1:-1,:])
            #df_all = df_all.iloc[-263570:,:]
            
            df_all = timerange1D(df_all)
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
            hourly_all = hourly_all.append(hourly_add.iloc[1:-1,:])
            hourly_all = timerange1D(hourly_all)
            _period='m15'
            daily_all = get_daily(hourly_all,TICKER_LIST)
                #del hourly_all
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
            features = scaling(features,scaler=_scaler)

            features = quantile(features,quantile_transform)

            _valid = _model.predict(features.drop(['Date','Symbol','Signal'],axis=1))[-1]

            _signal = df_all.Signal[-1]

            print('\nTest sur la bougie',features.index[-1])
            if _valid == 1 and _signal == 1 :
                buy(_period)

            elif _valid == 1 and _signal == -1 :
                sell(_period)

            else:
                print(col.Fore.BLUE,'\nNO SIGNAL FOR',col.Fore.YELLOW,x,'\n',col.Style.RESET_ALL)
            
            print('Reset of df_all')

            df_all = df_all[['Close','CloseAsk','CloseBid','High','HighAsk','HighBid','Low','LowAsk','LowBid','Open','OpenAsk','OpenBid','Symbol','Date']]  

