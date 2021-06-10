############################################
########### CONNEXION API FXCM #############
############################################


############################
######## FONCTIONS ########
###########################


def conX(TOKEN,server):
    global con
    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error',server=server)
    if con.is_connected() == True:
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
    else:
        print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
    return(con)



############################
######## LIBRAIRIES ########
############################
print('Importing Librairies...')
import colorama as col
import fxcmpy
import socketio
import pyttsx3

engine = pyttsx3.init()

print('Librairies imported\n')

engine.say("librairie loaded")
engine.runAndWait()
print('Prêt')


if __name__ == '__main__':
    TOKEN = '79f83cbff13d296eb6d9b6c1ed6dccd768ef925a'
    server = 'demo'
    user_id = 'D261219577'
    compte = '01215060'
    password = '3877'

    ########################
    ### CONNEXION A L'API ###
    #########################

    ___Author___='LumberJack Jyss'
    print('Global Optimized LumberJack Environment Motor for FOR_EX\nLumberJack Jyss 5780(c)')
    print(col.Fore.BLUE,'°0Oo_D.A.G._26_oO0°')
    print(col.Fore.YELLOW,col.Back.BLUE,'--- MEGA BASE MAKER FXCM v0.1 ---',col.Style.RESET_ALL)

    print('')
    engine.say(" Initialisation du Mega Base Maker v0.10")
    engine.say("Connexion du Gaulem hà la Péh e")
    engine.runAndWait()


    try:
        con = cf.conX(TOKEN,server)
        con.is_connected() == True
        print(col.Fore.GREEN+'Connexion établie'+col.Style.RESET_ALL)
        print('Compte utilisé : ',con.get_account_ids())
        engine.say("Connected")
        engine.runAndWait()
    except:
        print(col.Fore.RED+'Connexion non établie'+col.Style.RESET_ALL)
        engine.say("Not Connected, sal rass de mor, pitun cé la merd")
        engine.say("vérifi ton internet, é relance le gaulèm")
        engine.runAndWait()
        print('')
        #os._exit(0)
        #con = cf.deconX()
        time.sleep(1)
        con = cf.conX(TOKEN,server)