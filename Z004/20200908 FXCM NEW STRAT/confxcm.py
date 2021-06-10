###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
##############################      C O N N E X I O N      F S X C M      A P I       #####################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'

import configparser
config = configparser.ConfigParser()

config.read('config.ini')

TOKEN = config.get('CONNEXION','TOKEN')
server = config.get('CONNEXION','server')
user_id = config.get('CONNEXION','user_id')
compte = config.get('CONNEXION','compte')
password = config.get('CONNEXION','password')


############################################
########### CONNEXION API FXCM #############
############################################


############################
######## FONCTIONS ########
###########################


def conX():
    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error',server=server)
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



############################
######## LIBRAIRIES ########
############################
print('Importing Librairies...')
import colorama as col
import fxcmpy
import socketio
import datetime as dt
import time
import pyttsx3

engine = pyttsx3.init()

print('Librairies imported\n')

engine.say("librairie loaded")
engine.runAndWait()
print('Prêt')

def con_process():

    #########################
    ### CONNEXION A L'API ###
    #########################

    ___Author___='LumberJack Jyss'
    print('Global Optimized LumberJack Environment Motor for FOR_EX\nLumberJack Jyss 5780(c)')
    print(col.Fore.BLUE,'°0Oo_D.A.G._26_oO0°')
    print(col.Fore.YELLOW,col.Back.BLUE,'--- Go!em inside ---',col.Style.RESET_ALL)

    print('')
    _t1 = dt.datetime.now()
    engine.say(" Initialisation du Gaulem")
    engine.say("Connexion du Gaulem hà la Péh e")
    engine.runAndWait()

    try:
        con.is_connected() == True
        
        engine.say("already Connected")
        engine.runAndWait()
        print(col.Fore.GREEN+'Connexion rétablie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
        print('')
        _t2 = dt.datetime.now()
        print("Temps d'excution du module",str((_t2 - _t1)))
        
    except:
        try:
            con = conX()
            con.is_connected() == True
            print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
            print('Compte utilisé : ',con.get_account_ids())
            engine.say("Connected")
            engine.runAndWait()
            _t2 = dt.datetime.now()
            print("Temps d'excution du module",str((_t2 - _t1)))
        except:
            print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
            engine.say("Not Connected, sal rass de mor, pitun cé la merd")
            engine.say("vérifi ton internet, é relance le gaulèm")
            engine.runAndWait()
            print('')
            _t2 = dt.datetime.now()
            print("Temps d'excution du module",str((_t2 - _t1)))
    return(con)



if __name__ == '__main__':
    
    from confxcm import *
    con = con_process()