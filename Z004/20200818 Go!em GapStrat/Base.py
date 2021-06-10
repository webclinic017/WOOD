import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import pandas as pd
import datetime as dt
from ibapi.order import *
from colorama import Fore, Back, Style
import os
import pyttsx3
engine = pyttsx3.init()

###########################
######### CLASSES #########
###########################

class MyWrapper(EWrapper):

    def nextValidId(self, orderId:int):
        #4 first message received is this one
        self.nextValidOrderId = orderId
        #5 start requests here
        self.start()

    def historicalData(self, reqId:int, bar: BarData):
        #7 data is received for every bar
        DATA.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        return(DATA)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        #8 data is finished
        #9 this is the logical end of your program
        app.disconnect()        

    def error(self, reqId, errorCode, errorString):
        # these messages can come anytime.
        print("Error. Id: " , reqId, " Code: " , errorCode , " Msg: " , errorString)

    def start(self):
        #6 request data, using fx since I don't have Japanese data
        try:
            app.reqHistoricalData(_id, _contract, '',_duration, _timeframe, "MIDPOINT", 1, 1, False, [])
        except:
            pass
        
    
############################
######## FONCTIONS ########
###########################

def stock_contract(symbol, secType='STK', exchange='SMART', currency='USD'):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = secType
    contract.exchange = exchange
    contract.currency = currency
    contract.exchange = "ISLAND"
    return (contract)

def scrap_base(DATA):
    df = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
    DATE = []
    OPEN = []
    HIGH = []
    LOW = []
    CLOSE = []
    VOLUME = []
    for x in range(len(DATA)):
        DATE.append(DATA[x][0])
        OPEN.append(DATA[x][1])
        HIGH.append(DATA[x][2])
        LOW.append(DATA[x][3])
        CLOSE.append(DATA[x][4])
        VOLUME.append(DATA[x][5])
    df['Date'] = DATE
    df['Open'] = OPEN
    df['High'] = HIGH
    df['Low'] = LOW
    df['Close'] = CLOSE
    df['Volume'] = VOLUME
    df = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
    DATE = []
    OPEN = []
    HIGH = []
    LOW = []
    CLOSE = []
    VOLUME = []
    for x in range(len(DATA)):
        DATE.append(str(DATA[x][0][:4]+'-'+DATA[x][0][4:6]+'-'+DATA[x][0][6:8]+' '+DATA[x][0][10:18]))
        OPEN.append(DATA[x][1])
        HIGH.append(DATA[x][2])
        LOW.append(DATA[x][3])
        CLOSE.append(DATA[x][4])
        VOLUME.append(DATA[x][5])
    df['Date'] = DATE
    df['Open'] = OPEN
    df['High'] = HIGH
    df['Low'] = LOW
    df['Close'] = CLOSE
    df['Volume'] = VOLUME
    if _timeframe == '1 D' :
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    elif _timeframe == '1 hour':
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
    df.to_csv(_name+_ticker+'.csv')


engine.say("Do you want to update the base?")
engine.runAndWait()
_answer = input('Voulez vous mettre à jour la base? (oui - non)')


if _answer == 'oui' or _answer == 'o' or _answer == 'OUI' or _answer == 'O' or _answer == 'yes' or _answer == 'YES' or _answer == 'y' or _answer == 'Y':
    engine.say("Loading the teeckers")
    engine.runAndWait()
    print('Sraping tickers')

    SECTOR = []
    TICKERS = []
    BLACKLIST = ['BRK.B','BF.B']
    df_constituents = pd.read_csv('constituents.csv')
    df_constituents = df_constituents.drop_duplicates(['Symbol'])
    try:
        df_constituents = df_constituents.drop(['Unnamed: 0'],axis=1)
    except:
        pass
    SECTOR = list(sorted(set(df_constituents.Sector)))        
    TICKERS = list(sorted(set(df_constituents.Symbol.to_list())))
    for x in BLACKLIST:
        TICKERS.remove(x)

    engine.say("Teeckers loaded")
    engine.runAndWait()

    _duration = '5 Y'
    _timeframe = '1 day'
    _name = 'BASE_DAILY/'
    try:
        os.mkdir(_name)
    except:
        pass
    
    _id = 0

    for _ticker in TICKERS:
        print(Fore.BLUE,'Scraping du ticker (daily)',Fore.YELLOW,_ticker,Style.RESET_ALL)
        try:
            _id += 1
            DATA = []
            _contract = stock_contract(_ticker)
            app = EClient(MyWrapper()) 
            app.connect("127.0.0.1", 7496, clientId=230)
            app.run()
            scrap_base(DATA)
        except:
            continue

    _duration = '5 Y'
    _timeframe = '1 hour'
    _name = 'BASE_HOURLY/'
    try:
        os.mkdir(_name)
    except:
        pass
    _id = 0

    for _ticker in TICKERS:
        print(Fore.BLUE,'Scraping du ticker (hourly)',Fore.YELLOW,_ticker,Style.RESET_ALL)
        try:
            _id += 1
            DATA = []
            _contract = stock_contract(_ticker)
            app = EClient(MyWrapper()) 
            app.connect("127.0.0.1", 7496, clientId=230)
            app.run()
            scrap_base(DATA)
        except:
            continue

        engine.say("Bases scraped")
        engine.runAndWait()

        app = EClient(MyWrapper()) #1 create wrapper subclass and pass it to EClient
        app.connect("127.0.0.1", 7496, clientId=101) #2 connect to TWS/IBG

        if app.isConnected() == True:
            print('Deconnexion en cours...')
            app.disconnect()
        time.sleep(1)
        
        if app.isConnected() == False:
            print('Deconnexion effectuée')

