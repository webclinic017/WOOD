def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
clearall()
print('Importing Librairies...')
import talib
import numpy as np
import pandas as pd
import pandas_datareader as web
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
import seaborn as sns
plt.style.use('seaborn')
import time
import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,roc_curve,confusion_matrix,classification_report
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print('Librairies imported')
print('')

___Author___='LumberJack Jyss'
print('Global Optimized LumberJack Environment Motor 55\nLumberJack Jyss 5779(c)')
print(Fore.BLUE,'°0Oo_D.A.G._26_oO0°')
print('BOOST SKAN 55 Version v5.55',Style.RESET_ALL)

LaDate = input('Date de DL - YYYY-MM-DD ')
try:
    os.mkdir('DL_'+LaDate)
except:
    pass

print('')
print('Sraping tickers')
constituents = pd.read_csv('New.csv')
print('Scrap -----> ok')
# PARAMETRES TEMPORELS INITIAUX
start =LaDate[:2]+str(int(LaDate[2:4])-4)+'-'+LaDate[5:]
end = LaDate

error = []


try :
    amorceur = pd.read_csv('DL_'+LaDate+'/compteur'+LaDate+'.csv')
    amorceur = amorceur.drop(['Unnamed: 0'],axis=1)
    amorce = constituents[constituents['Symbol']==amorceur.iloc[-1,0]].index[-1]+1
    compteur = pd.read_csv('DL_'+LaDate+'/compteur'+LaDate+'.csv')
except:
    amorce = 0
    compteur = pd.DataFrame(columns = ['Symb.','Name','Sector','Precision_up','Precision_down'])
       
# SCRAPING DES DONNES BRUTES
def scrap_data(ticker,start,end):
    df = web.DataReader(ticker,'yahoo',start,end)
    df = df.drop(['Close'],axis=1)
    df['Close'] = df['Adj Close']
    df = df.drop(['Adj Close'],axis = 1)
    df = df[-820:]
    return(df)

def prepa_data(df):
    rsi = talib.RSI(df['Close'],timeperiod=14)
    stoc_slowk, stoc_slowd = talib.STOCH(df['High'],df['Low'],df['Close'])
    upper, middle, lower =  talib.BBANDS(df['Close'], timeperiod=9, nbdevup=2, nbdevdn=2,matype=0)
    sma5 = talib.SMA(df['Close'],timeperiod=5)
    sma8 = talib.SMA(df['Close'],timeperiod=8)
    sma10 = talib.SMA(df['Close'],timeperiod=10)
    sma12 = talib.SMA(df['Close'],timeperiod=12)
    sma15 = talib.SMA(df['Close'],timeperiod=15)
    sma30 = talib.SMA(df['Close'],timeperiod=30)
    sma35 = talib.SMA(df['Close'],timeperiod=35)
    sma40 = talib.SMA(df['Close'],timeperiod=40)
    sma45 = talib.SMA(df['Close'],timeperiod=45)
    sma50 = talib.SMA(df['Close'],timeperiod=50)
    atr = talib.ATR(df['High'],df['Low'],df['Close'],timeperiod=10)
    delta5_8 = sma5 - sma8
    delta8_10 = sma8 - sma10
    delta10_12 = sma10 - sma12
    delta12_15 = sma12 - sma15
    delta15_30 = sma15 - sma30
    delta30_35 = sma30 - sma35
    delta35_40 = sma35 - sma40
    delta40_45 = sma40 - sma45
    delta45_50 = sma45 - sma50
    bbdelta = upper - middle
    price_bolup = df['Close'] - lower
    price_bolow = df['Close'] - upper
    Ema = talib.EMA(df['Close'],timeperiod=20)
    KC_High = Ema + 2*atr
    KC_Low = Ema - 2*atr
    aroondown, aroonup = talib.AROON(df['High'], df['Low'], timeperiod=9)
    aroon = aroonup - aroondown 
    rsi30_list = []
    rsi70_list = []
    for i in range(0,df.shape[0]):
        rsi70_list.append(70 - rsi[i])
        rsi30_list.append(rsi[i] - 30)
        
    varop_spy = df['Open'] - df['Close']
    varhl_spy = df['High'] - df['Low']
    df['Varop_Spy'] = varop_spy
    df['Varhl_spy'] = varhl_spy
    df['RSI'] = rsi
    df['70 - RSI'] = np.array(rsi70_list)
    df['RSI - 30'] = np.array(rsi30_list)
    df['BBD_Delta_Up'] = bbdelta
    df['delta5_8'] = delta5_8
    df['delta8_10'] = delta8_10
    df['delta10_12'] = delta10_12
    df['delta12_15'] = delta12_15
    df['delta15_30'] = delta15_30
    df['delta30_35'] = delta30_35
    df['delta35_40'] = delta35_40
    df['delta40_45'] = delta40_45
    df['delta45_50'] = delta45_50
    df['Stoc_Slowk'] = stoc_slowk
    df['Stoc_Slowd'] = stoc_slowd
    df['KC_High'] = KC_High
    df['KC_Low'] = KC_Low
    df['upper'] = upper
    df['lower'] = lower
    df['var_bollup_kchigh'] = upper-KC_High
    df['var_bolllow_kclow'] = lower-KC_Low
    df['Aroon Up'] = aroonup
    df['Aroon Down'] = aroondown
    df['Delta Aroon'] = aroon
    up = []
    down = []
    df = df.dropna()
    df = boost(df)
    df['%Futur'] = ((df['Close.S']-df['Close']) *100) / (df['Close'])
    df['%Futur2'] = ((df['Close.S2']-df['Close']) *100) / (df['Close'])
    for i in range(0,df.shape[0]-5):
        if df.iloc[i]['%Futur'] > 0.5 :
            up.append(1)
            down.append(0)
        elif df.iloc[i]['%Futur'] < -0.5:
            up.append(0)
            down.append(1)
        else:
            up.append(0)
            down.append(0)
    up.append(0)
    down.append(0)
    up.append(0)
    down.append(0)
    up.append(0)
    down.append(0)
    up.append(0)
    down.append(0)
    up.append(0)
    down.append(0)
    
    
    df['target_up'] = up  
    df['target_down'] = down 
    #df = df.dropna()
    return(df)

def boost(df):
    X = df.copy()
    X = X.drop(['Close'],axis=1)
    X['Close'] = df['Close']
    y = X.iloc[:,-1]
    Xtrain = X.iloc[:-2,:-1]
    Xtest = X.iloc[-2:-1,:-1]
    yshift = y.shift(-1)
    ytrain = yshift.iloc[:-2]
    ytest = yshift.iloc[-2:-1]

    model = xgb.XGBRegressor(n_estimators=20000, learning_rate=1, gamma=1, subsample=1, colsample_bytree=1, max_depth=100,objective='reg:squarederror')

    model.fit( Xtrain, ytrain, early_stopping_rounds=150, eval_set=[(Xtest, ytest)], verbose=0)

    ytrain_pred = model.predict(Xtrain)

    y_pred = model.predict(Xtest)

    pred = model.predict(X.iloc[:,:-1])

    df['Close.S'] = pred
    df['Close.S2'] = df['Close.S']
    df = df.dropna()
    return(df)

    
def deep_learning(df):
    X = df.iloc[:,:-4]
    y_up = df.iloc[:,-2].values
    y_down = df.iloc[:,-1].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y_up = np.array(y_up).reshape(-1,1)
    y_down = np.array(y_down).reshape(-1,1)

    Xtrain = X[:bloc1,:]
    Xtest = X[bloc1:,:]
    ytrain_up = y_up[:bloc1,:]
    ytest_up = y_up[bloc1:,:]
    ytrain_down = y_down[:bloc1,:]
    ytest_down = y_down[bloc1:,:]

    seed = 770
    np.random.seed(seed)

    ytrain_up = ytrain_up.reshape(ytrain_up.shape[0],)
    ytrain_down = ytrain_down.reshape(ytrain_down.shape[0],)

    Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1])

    model_up = Sequential()
    # Add an input layer 
    model_up.add(Dense(50, activation='relu'))
    # Add one hidden layer 
    model_up.add(Dense(23, activation='relu'))
    # Add an output layer 
    model_up.add(Dense(1, activation='sigmoid'))

    model_down = Sequential()
    # Add an input layer 
    model_down.add(Dense(50, activation='relu'))
    # Add one hidden layer 
    model_down.add(Dense(23, activation='relu'))
    # Add an output layer 
    model_down.add(Dense(1, activation='sigmoid'))

    model_up.compile(loss='binary_crossentropy',
                  optimizer='adam', #rmsprop
                  metrics=['accuracy','mse'])
    
              

    history_up = model_up.fit(Xtrain, ytrain_up,epochs=280, batch_size=8, verbose=0)
    
    model_down.compile(loss='binary_crossentropy',
                  optimizer='adam', #rmsprop
                  metrics=['accuracy','mse'])

    history_down = model_down.fit(Xtrain, ytrain_down,epochs=280, batch_size=8, verbose=0)
    

    train_acc_up = model_up.evaluate(Xtrain, ytrain_up,verbose=1)
    train_acc_down = model_down.evaluate(Xtrain, ytrain_down,verbose=1)

    yhat_up = model_up.predict_classes(Xtest)
    yhat_down = model_down.predict_classes(Xtest)

    score_up = model_up.evaluate(Xtest, ytest_up,verbose=1)
    score_down = model_down.evaluate(Xtest, ytest_down,verbose=1)

    predict_up = model_up.predict(Xtest)
    predict_down = model_down.predict(Xtest)

    accuracy_up = accuracy_score(ytest_up, yhat_up)
    accuracy_down = accuracy_score(ytest_down, yhat_down)

    # La précision permet de mesurer la capacité du modèle à refuser résultats non-pertinents : vrais_positifs/(vrais_positifs+faux_positifs)
    precision_up = precision_score(ytest_up, yhat_up)  
    precision_down = precision_score(ytest_down, yhat_down) 


    # Recall : (vrai_positifs/(vrais_positifs+faux_négatifs))
    recall_up = recall_score(ytest_up, yhat_up) 
    recall_down = recall_score(ytest_down, yhat_down) 


    resultats = pd.DataFrame()
    resultats['Date'] = df.index[bloc1:]
    resultats.index= df.index[bloc1:]
    resultats['Move Up'] = yhat_up
    resultats['Confiance up'] = (predict_up)*100
    resultats['Move Down'] = yhat_down
    resultats['Confiance Down'] = (predict_down)*100
    resultats['Actual'] = df.iloc[bloc1:]['Close']
    resultats['Actual.S'] = df.iloc[bloc1:]['Close.S']
    open_S = df['Open'].shift(-1)
    resultats['Open.S'] = open_S.iloc[bloc1:]
    dmp_cp=[]
    dmp_cp = ((resultats['Confiance up']-resultats['Confiance Down'])/(resultats['Confiance up']+resultats['Confiance Down'])*100)
    resultats['DMP_CP'] = dmp_cp
    
    return(resultats,precision_up,precision_down,model_up,model_down,scaler)


def save_model(model_up,model_down):
    savename = 'DL_'+LaDate+'/Save_'+ticker
    # serialize model to YAML
    model_up_yaml = model_up.to_yaml()
    model_down_yaml = model_down.to_yaml()
    with open(savename+"_up.yaml", "w") as yaml_file:
        yaml_file.write(model_up_yaml)
    with open(savename+"_down.yaml", "w") as yaml_file:
        yaml_file.write(model_up_yaml)
    # serialize weights to HDF5
    model_up.save_weights(savename+"_up.h5")
    model_down.save_weights(savename+"_down.h5")
    
########################
#### MAIN SKAN55 #######
########################
ticker_list = compteur['Symb.'].tolist()
name_list = compteur['Name'].tolist()
sector_list = compteur['Sector'].tolist()
prec_up_list = compteur['Precision_up'].tolist()
prec_down_list = compteur['Precision_down'].tolist()

tmps55=time.time()
try:
    print(Fore.BLUE,'Deeping in blue from ',ticker_list[-1],Style.RESET_ALL)
except:
    print(Fore.BLUE,'Deeping in blue from ','A',Style.RESET_ALL)

for loop in range(amorce,amorce+10):
#for loop in range(amorce,len(constituents)):
    
    try:

        ticker = (constituents.iloc[loop]['Symbol'])
        name = constituents.iloc[loop]['Name']
        sector = constituents.iloc[loop]['Sector']
        
        global delta,bloc1,bloc2
        tmps1=time.time()
        df = scrap_data(ticker,start,end)

        tmps2=round(time.time()-tmps1,2)
        delta = round(df.shape[0])
        bloc1 = round(delta*0.80)
        bloc2 = delta - bloc1
       
        df = prepa_data(df)

        resultats,precision_up,precision_down,model_up,model_down,scaler = deep_learning(df)

        if (precision_up*100) < 69 or (precision_down*100) < 69:
            resultats,precision_up,precision_down,model_up,model_down,scaler = deep_learning(df)

        if (precision_up*100) < 69 or (precision_down*100) < 69:
            resultats,precision_up,precision_down,model_up,model_down,scaler = deep_learning(df)

        if (precision_up*100) < 69 or (precision_down*100) < 69:
            print('Test precision raté 3 fois pour le ticker ',ticker)
            continue

        ticker_list.append(ticker)
        name_list.append(name)
        sector_list.append(sector)
        prec_up_list.append(round(precision_up*100,2))
        prec_down_list.append(round(precision_down*100,2))
        save_model(model_up, model_down)

        print('Le ',Fore.BLUE,'Deep Learning',Style.RESET_ALL ,'de ',Fore.YELLOW,ticker,Style.RESET_ALL,' a été effecué avec succès. Les modèles ont été sauvegardés')

    except:
        print(Fore.RED,'Problème loop : ',loop,Style.RESET_ALL)
        error.append((loop,ticker))
         
        continue

print(Fore.YELLOW,Back.BLUE,'Longueur des listes pour vérification : ',len(ticker_list),len(name_list),len(sector_list),Style.RESET_ALL)

compteur = pd.DataFrame(columns = ['Symb.','Name','Sector'])

compteur['Symb.'] = ticker_list

compteur['Name'] = name_list

compteur['Sector'] = sector_list

compteur['Precision_up'] = prec_up_list

compteur['Precision_down'] = prec_down_list

compteur.to_csv('DL_'+LaDate+'/compteur'+LaDate+'.csv')

print(Fore.YELLOW,Back.MAGENTA,Style.DIM,'PASSAGE FINI!!!!!!',Style.RESET_ALL)
tmps2=round(time.time()-tmps55,2)
print ("Job done in = %f" %tmps2,'seconds')