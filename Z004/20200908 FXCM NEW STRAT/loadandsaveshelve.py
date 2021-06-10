###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
#############################      S A U V E G A R D E      D E S      T A B L E S        #################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
__author__ = 'LumberJack Jyss'
__copyright__ = '(c) 5780'

import os
import shelve

if os.path.isdir('TABLES/') == False:
    print('Création du dossier TABLES')
    os.mkdir('TABLES')

if os.path.isdir('VARIABLES/') == False:
    print('Création du dossier VARIABLES')
    os.mkdir('VARIABLES')


if os.path.isfile('TABLES/tables') == False:
    TABLE = {}


def save_data(TABLE):
    print('Sauvegarde de la table')
    t = shelve.open('TABLES/tables')
    t['TABLES'] = TABLE
    t.close()
    return()

def load_data():
    t = shelve.open('TABLES/tables')
    for key,val in t.items():
        exec(key + '=val') 
    TABLE = t['TABLES']
    t.close()
    return(TABLE)

def save_var(name,var):
    print('Sauvegarde de la variable '+name)
    t = shelve.open('VARIABLES/'+name)
    t[name] = var
    t.close()
    return()

def load_var(name):
    global key
    t = shelve.open('VARIABLES/'+name)
    for key,val in t.items():
        exec(key + '=val')
    t.close()
    return(key,val)


if __name__ == "__main__":
    
    from loadandsaveshelve import *
    save_data(TABLE)
    TABLE = load_data()
