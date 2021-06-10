###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
############################      M A K E      L I S T      O F      T I C K E R S         ################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'

from os import listdir
from os.path import isfile, join
import configparser

config = configparser.ConfigParser()

config.read('config.ini')
_path = config.get('PATH','_path') # 'Base_Clean/'


def scrap_tickers(_path):
    print(_path)
    TICKERS = list(set(sorted([f[:3]+'/'+f[3:6] for f in listdir(_path) if isfile(join(_path, f))])))
    try:
        TICKERS.remove('.DS/_St')
    except:
        pass
    for x in TICKERS:
        if '.' in x:
            TICKERS.remove(x)
            print('problème rencontré avec le ticker',x)
    return(TICKERS)

if __name__ == "__main__":
    pass