from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order_condition import Create, OrderCondition
from ibapi.order import *
import pandas as pd
import time
import datetime as dt
import pyttsx3
engine = pyttsx3.init()
import threading
from colorama import Fore, Back, Style
import os


###########################
######### CLASSES #########
###########################

class IBapi(EWrapper, EClient):
    def __init__(self):
	    EClient.__init__(self, self)
	    self.data = [] #Initialize variable to store candle
    
    def tickPrice(self, reqId, tickType, price, attrib):
	    if tickType == 2 and reqId == 1:
		    print('The current ask price is: ', price)
    
    def nextValidId(self, orderId: int):
	    super().nextValidId(orderId)
	    self.nextorderId = orderId
	    print('The next valid order id is: ', self.nextorderId)

    def orderStatus(self, orderId, status, filled, remaining, avgFullPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
	    print('orderStatus - orderid:', orderId, 'status:', status, 'filled', filled, 'remaining', remaining, 'lastFillPrice', lastFillPrice)
	
    def openOrder(self, orderId, contract, order, orderState):
	    print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange, ':', order.action, order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
	    print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency, execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)
    
    def error(self, reqId, errorCode, errorString):
        if errorCode == 202:
            print('order canceled')


############################
######## FONCTIONS ########
###########################

def sector():
    SECTOR = []
    TICKERS = []
    df_constituents = pd.read_csv('constituents.csv')
    print('shape : ',df_constituents.shape)
    df_constituents = df_constituents.drop_duplicates(['Symbol'])
    try:
        df_constituents = df_constituents.drop(['Unnamed: 0'],axis=1)
    except:
        pass
    print('shape : ',df_constituents.shape) 
    SECTOR = list(sorted(set(df_constituents.Sector)))
    
    TICKERS = df_constituents.Symbol.to_list()

    print('Scrap -----> ok')

    _today = dt.datetime.now().date().strftime("%Y-%m-%d")
    _start = (dt.datetime.now().date() - dt.timedelta(days=756)).strftime("%Y-%m-%d")
    return(SECTOR,TICKERS,df_constituents)

def run_loop():
    app.run()

#Function to create FX Order contract
def FX_order(symbol):
    contract = Contract()
    contract.symbol = symbol[:3]
    contract.secType = 'CASH'
    contract.exchange = 'IDEALPRO'
    contract.currency = symbol[3:]
    return contract

def stock_contract(symbol, secType='STK', exchange='SMART', currency='USD'):
	''' custom function to create stock contract '''
	contract = Contract()
	contract.symbol = symbol
	contract.secType = secType
	contract.exchange = exchange
	contract.currency = currency
	return contract

def contractDetails(self, reqId: int, contractDetails):
	    self.contract_details[reqId] = contractDetails

def get_contract_details(self, reqId, contract):
    self.contract_details[reqId] = None
    self.reqContractDetails(reqId, contract)
    #Error checking loop - breaks from loop once contract details are obtained
    for err_check in range(50):
        if not self.contract_details[reqId]:
            time.sleep(0.1)
        else:
            break
    #Raise if error checking loop count maxed out (contract details not obtained)
    if err_check == 49:
        raise Exception('error getting contract details')
    #Return contract details otherwise
    return app.contract_details[reqId].contract 



engine.say("Do you want to update the base?")
engine.runAndWait()
_answer = input('Voulez vous mettre à jour la base? (oui - non)')

if _answer == 'oui' or _answer == 'o' or _answer == 'OUI' or _answer == 'O' or _answer == 'yes' or _answer == 'YES' or _answer == 'y' or _answer == 'Y':
    engine.say("Loading the teeckers")
    engine.runAndWait()
    print('Sraping tickers')

    try:
        os.mkdir('BASE')
    except:
        pass

    SECTOR,TICKERS,df_constituents = sector()

    for _sector in SECTOR:
        globals()['LIST_%s' %_sector] = df_constituents[df_constituents.Sector == _sector].Symbol.to_list()

    for _sector in SECTOR:
        for _ticker in globals()['LIST_%s' %_sector]:
            globals()['df_%s' %_ticker] = pd.DataFrame()

    engine.say("Teeckers loaded")
    engine.runAndWait()

    app = IBapi()
    app.connect('127.0.0.1', 7496, 123)

    time.sleep(1)
    if app.isConnected() == True:
        print('Connexion établie sur compte LIVE')
    else:
        print('Problème de connexion')

    #Start the socket in a thread
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()

    time.sleep(1) #Sleep interval to allow time for connection to server
    _id = 0
    for _ticker in TICKERS:
        print(Fore.BLUE,'\rScraping du ticker',Fore.YELLOW,_ticker,Style.RESET_ALL,end='',flush=True)
        _id += 1
        _contract = stock_contract(_ticker, secType='STK', exchange='SMART', currency='USD')

        #Request historical candles
        app.reqHistoricalData(_id, _contract, '', '6 M', '1 day', 'MIDPOINT', 0, 1, False, [])

        time.sleep(5) #sleep to allow enough time for data to be returned

        globals()['df_%s' %_ticker] = pd.DataFrame(app.data, columns=['DateTime', 'Open', 'High', 'Low', 'Close','Volume'])
        globals()['df_%s' %_ticker]['Date'] = pd.to_datetime(globals()['df_%s' %_ticker]['DateTime'],unit='s') 
        globals()['df_%s' %_ticker].to_csv('BASE/'+_ticker+'.csv')  

    app.disconnect()

    if app.isConnected() == True:
        print('Connexion établie')
    else:
        print('Deconnexion effectuée') 
