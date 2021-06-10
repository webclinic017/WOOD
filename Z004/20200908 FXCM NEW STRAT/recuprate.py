###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
############################      R E C U P E R A T I O N     D E S      R A T E S        #################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'

import colorama as col
import datetime as dt
import multiprocessing.dummy as mp 
import time
import pyttsx3
import configparser
config = configparser.ConfigParser()

config.read('config.ini')

TICKERS = config.get('TICKERS','TICKERS')
TICKERS = TICKERS.split(',')

engine = pyttsx3.init()

RATE = []


def recup_rate(x):
    if x[4:] == 'USD':
        _rate = (x , 1)

    elif x[:3] == 'USD':
        con.subscribe_market_data(x)
        _rate = (x , con.get_last_price(x).Bid)
        con.unsubscribe_market_data(x)
    elif x[:3] != 'USD' and x[4:] != 'USD':
        try:
            con.subscribe_market_data(x[4:]+'/USD')
            _rate = (x , con.get_last_price(x[4:]+'/USD').Bid)
            con.unsubscribe_market_data(x[4:]+'/USD')
            _flag = 1
        except:
            con.subscribe_market_data('USD/'+x[4:])
            _rate = (x , con.get_last_price('USD/'+x[4:]).Bid)
            con.unsubscribe_market_data('USD/'+x[4:])
            _flag = -1
    else:
        _rate = 'NA'

    print(col.Fore.MAGENTA,'Le rate du ticker',x,'est à ',_rate,col.Style.RESET_ALL)
    RATE.append(_rate)
    return(RATE)

def run_recuprate():
    _t1 = dt.datetime.now()
    print('Début des opérations')
    p=mp.Pool(os.cpu_count())
    p.map(recup_rate(),TICKERS) 
    p.close()
    p.join()
    print("\n\n ===> Réindexation des bases terminée. Tout est nettoyé et prêt à l'utilisation")
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return(RATE)


if __name__ == "__main__":
    pass
