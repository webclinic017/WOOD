
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
##################################      S C R A P      C S V      O N      F X C M       ##################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'

import csv
import pandas as pd
import colorama as col
import configparser
config = configparser.ConfigParser()

config.read('config.ini')
_period1 = config.get('TIMEFRAME','_period1') # 'm5'
_period2 = config.get('TIMEFRAME','_period2') # 'H1'
_period3 = config.get('TIMEFRAME','_period3') # 'D1'
_path1 = config.get('PATH','_path1') # 'Base/'
_path2 = config.get('PATH','_path2') # 'Base_Clean'
_path3 = config.get('PATH','_path3') # 'Base_Input'
_path = config.get('PATH','_path') # 'Base_Input'
TICKERS = config.get('TICKERS','TICKERS')
TICKERS = TICKERS.split(',')

def scrap_csv(x):
    config.read('config.ini')
    _period1 = config.get('TIMEFRAME','_period1') # 'm5'
    _period2 = config.get('TIMEFRAME','_period2') # 'H1'
    _period3 = config.get('TIMEFRAME','_period3') # 'D1'
    _path1 = config.get('PATH','_path1') # 'Base/'
    _path2 = config.get('PATH','_path2') # 'Base_Clean'
    _path3 = config.get('PATH','_path3') # 'Base_Input'
    _path = config.get('PATH','_path') # 'Base_Input'
    TICKERS = config.get('TICKERS','TICKERS')
    TICKERS = TICKERS.split(',')
    if '.' in x:
        TICKERS.remove(x)
        print('Remove',x)
    print('\r',col.Fore.BLUE,'Scraping des données OHLC pour le ticker',col.Fore.YELLOW,x,col.Style.RESET_ALL,' --- ',end='',flush=True)
    #with open('Base/'+x.replace('/','')+'_'+_period+'_BidAndAsk.csv', 'r') as csvfile:
    #    dialect = csv.Sniffer().sniff(csvfile.readline())
    #    _delimiter = dialect.delimiter
    globals()['df1_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period1+'.csv')#,delimiter=_delimiter)
    globals()['df2_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period2+'.csv')#,delimiter=_delimiter)
    globals()['df3_%s' %x.replace('/','')] = pd.read_csv(_path+x.replace('/','')+_period3+'.csv')#,delimiter=_delimiter)

    return(globals()['df1_%s' %x.replace('/','')],globals()['df2_%s' %x.replace('/','')],\
        globals()['df3_%s' %x.replace('/','')])

if __name__ == "__main__":
    pass