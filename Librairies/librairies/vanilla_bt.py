import colorama as col
import datetime as dt

__author__ = 'LumberJack'

##### DAG26 5781 (c) #####
#####
##### _df : Dataframe doit contenir colonne Signal [-1,0,1] /!\ DEJA SHIFTEE .shift(-1) /!\ pour tester la robustesse d'un signal à prédir.
#####
##### _tp : Take Profit [0:1]
#####
##### _sl : Stop Loss [0:1]
#####
##### _te : Exit en nombre de bougis
#####
##### iv : Basé sur le signal inverse
#####
##### BTs disponiples : _bt_te(X,_te), bt_iv, bt_tpsl(X,_tp,_sl), ==> bt_tpsl_grid(X) <== grid pour tester un pool de tp/sl

def bt_tpsl(X,_tp,_sl):
    _size = 50000
    _capital = 250000
    _open_buy = 0
    _open_sell = 0
    _price_buy = 0
    _price_sell = 0
    _nb_trade = 0
    _start_buy = 0
    _start_sell = 0
    _cpt_buy = 0
    _cpt_sell = 0 
    _pnl = 0
    _win = 0
    _los = 0
    PNL = []

    for i in range(len(X)):

        # Brute Closing
        if i == len(X)-1 :
            if _open_buy == 1:
                print("fermeture long en l'air")
                _pnl = _size * (X.Close[i] - _price_buy)
                _nb_trade += 1
                _open_buy = 999
                _open_sell = 999
                _price_buy = 0
                PNL.append(_pnl)
                if _pnl > 0:
                    _win += 1
                else:
                    _los += 1

            if _open_sell == 1:
                print("fermeture short en l'air")
                _pnl =  - _size * (X.Close[i] - _price_sell)
                _nb_trade += 1
                _open_sell = 999
                _open_buy = 999
                _price_sell = 0
                PNL.append(_pnl)
                if _pnl > 0:
                    _win += 1
                else:
                    _los += 1

        # Open Long
        if X.Signal[i] == 1 and _open_buy == 0 and _open_sell == 0:
            _price_buy = X.Close[i]
            _open_buy = 1
            _start_buy = i
            _cpt_buy += 1


        # TP Short Close
        if (X.Close[i] - _price_sell)/_price_sell <= - _tp and _open_buy == 0 and _open_sell == 1:
            _pnl =  - _size * (X.Close[i] - _price_sell)
            _nb_trade += 1
            _open_sell = 0
            _price_sell = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

        # SL Close Short
        if (X.Close[i] - _price_sell)/_price_sell >=  _sl and _open_buy == 0 and _open_sell == 1:
            _pnl =  - _size * (X.Close[i] - _price_sell)
            _nb_trade += 1
            _open_sell = 0
            _price_sell = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

        # Open Short
        if X.Signal[i] == -1 and _open_buy == 0 and _open_sell == 0:
            _price_sell = X.Close[i]
            _open_sell = 1
            _start_sell = i
            _cpt_sell += 1

        # TP Close Long
        if (X.Close[i] - _price_buy)/_price_buy >=  _tp and _open_sell == 0 and _open_buy == 1:
            _pnl = _size * (X.Close[i] - _price_buy)
            _nb_trade += 1
            _open_buy = 0
            _price_buy = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

        # SL Close Long
        if (X.Close[i] - _price_buy)/_price_buy <= - _sl and _open_sell == 0 and _open_buy == 1:
            _pnl = _size * (X.Close[i] - _price_buy)
            _nb_trade += 1
            _open_buy = 0
            _price_buy = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

    try:
        print('Winner Ratio :',round((_win/(_win+_los))*100,2),'%')
    except:
        if _win == 0:
            print('Aucun Winner')
        else:
            print('Probleme avec le Ratio')
    try:
        print('Profit Factor :',col.Fore.BLUE,round(abs(sum(list(filter(lambda x: x > 0, PNL))) / sum(list(filter(lambda x: x <= 0, PNL)))),2),'%',col.Style.RESET_ALL)
    except:
        print(col.Fore.RED,'PNL à O',col.Style.RESET_ALL)
    print(col.Fore.MAGENTA,'Période du',X.index[0],'au ',X.index[-1],col.Style.RESET_ALL)
    print('Nombre de trades :',_nb_trade)
    print('Nombre de posistions non fermées',(_cpt_buy+_cpt_sell)-_nb_trade)
    print('Captital initial :',_capital)
    print('Taille des positions :',_size)
    if sum(PNL) > 0:
        print('Gain :',col.Fore.GREEN,sum(PNL),col.Style.RESET_ALL)
    else:
        print('Gain :',col.Fore.RED,sum(PNL),col.Style.RESET_ALL)
    print('Capital Final :',_capital + sum(PNL))
    return()

def bt_te(X,_te):
    _size = 50000
    _capital = 250000
    _open_buy = 0
    _open_sell = 0
    _price_buy = 0
    _price_sell = 0
    _nb_trade = 0
    _start_buy = 0
    _start_sell = 0
    _cpt_buy = 0
    _cpt_sell = 0 
    _pnl = 0
    _win = 0
    _los = 0
    PNL = []

    for i in range(len(X)):


        if i == len(X)-1 :
            if _open_buy == 1:
                print("fermeture long en l'air")
                _pnl = _size * (X.Close[i] - _price_buy)
                _nb_trade += 1
                _open_buy = 999
                _open_sell = 999
                _price_buy = 0
                PNL.append(_pnl)
                if _pnl > 0:
                    _win += 1
                else:
                    _los += 1

            if _open_sell == 1:
                print("fermeture shirt en l'air")
                _pnl =  - _size * (X.Close[i] - _price_sell)
                _nb_trade += 1
                _open_sell = 999
                _open_buy = 999
                _price_sell = 0
                PNL.append(_pnl)
                if _pnl > 0:
                    _win += 1
                else:
                    _los += 1


        if X.Signal[i] == 1 and _open_buy == 0 and _open_sell == 0:
            _price_buy = X.Close[i]
            _open_buy = 1
            _start_buy = i
            _cpt_buy += 1

        if i - _start_sell == _te and _open_buy == 0 and _open_sell == 1:
            _pnl =  - _size * (X.Close[i] - _price_sell)
            _nb_trade += 1
            _open_sell = 0
            _price_sell = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

        if X.Signal[i] == -1 and _open_buy == 0 and _open_sell == 0:
            _price_sell = X.Close[i]
            _open_sell = 1
            _start_sell = i
            _cpt_sell += 1

        
        if i - _start_buy == _te and _open_sell == 0 and _open_buy == 1:
            _pnl = _size * (X.Close[i] - _price_buy)
            _nb_trade += 1
            _open_buy = 0
            _price_buy = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1
    try:
        print('Winner Ratio :',round((_win/(_win+_los))*100,2),'%')
    except:
        if _win == 0:
            print('Aucun Winner')
        else:
            print('Probleme avec le Ratio')
    try:
        print('Profit Ratio :',round(abs(sum(list(filter(lambda x: x > 0, PNL))) / sum(list(filter(lambda x: x <= 0, PNL)))),2),'%')
    except:
        print('PNL à O')
    print(col.Fore.MAGENTA,'Période du',X.index[0],'au ',X.index[-1],col.Style.RESET_ALL)
    print('Nombre de trades :',_nb_trade)
    print('Nombre de posistions non fermée',(_cpt_buy+_cpt_sell)-_nb_trade)
    print('Captital initial :',_capital)
    print('Taille des positions :',_size)
    print('Gain :',sum(PNL))
    print('Capital Final :',_capital + sum(PNL))
    return()

def bt_iv(X):
    _size = 50000
    _capital = 250000
    _open_buy = 0
    _open_sell = 0
    _price_buy = 0
    _price_sell = 0
    _nb_trade = 0
    _cpt_buy = 0
    _cpt_sell = 0
    _pnl = 0
    _win = 0
    _los = 0
    PNL = []

    for i in range(len(X)):
        if X.Signal[i] == 1 and _open_buy == 0 and _open_sell == 0:
            _price_buy = X.Close[i]
            _open_buy = 1
            _cpt_buy += 1

        if X.Signal[i] == 1 and _open_buy == 0 and _open_sell == 1:
            _pnl =  - _size * (X.Close[i] - _price_sell)
            _nb_trade += 1
            _open_sell = 0
            _price_sell = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

        if X.Signal[i] == -1 and _open_buy == 0 and _open_sell == 0:
            _price_sell = X.Close[i]
            _open_sell = 1
            _cpt_sell += 1

        
        if X.Signal[i] == -1 and _open_sell == 0 and _open_buy == 1:
            _pnl = _size * (X.Close[i] - _price_buy)
            _nb_trade += 1
            _open_buy = 0
            _price_buy = 0
            PNL.append(_pnl)
            if _pnl > 0:
                _win += 1
            else:
                _los += 1

    try:
        print('Winner Ratio :',round((_win/(_win+_los))*100,2),'%')
    except:
        if _win == 0:
            print('Aucun Winner')
        else:
            print('Probleme avec le Ratio')
    try:
        print('Profit Ratio :',round(abs(sum(list(filter(lambda x: x > 0, PNL))) / sum(list(filter(lambda x: x <= 0, PNL)))),2),'%')
    except:
        print('PNL à O')
    print(col.Fore.MAGENTA,'Période du',X.index[0],'au ',X.index[-1],col.Style.RESET_ALL)
    print('Nombre de trades :',_nb_trade)
    print('Nombre de posistions non fermée',(_cpt_buy+_cpt_sell)-_nb_trade)
    print('Captital initial :',_capital)
    print('Taille des positions :',_size)
    print('Gain :',sum(PNL))
    print('Capital Final :',_capital + sum(PNL))
    return()


def bt_tpsl_grid(X):

    _t1 = dt.datetime.now()
    
    
    TP = [.1,0.09,0.08,0.07,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.0009]
    SL = [.1,0.09,0.08,0.07,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.0009]


    for _tp in TP:
        for _sl in SL:
            _size = 50000
            _capital = 250000
            _open_buy = 0
            _open_sell = 0
            _price_buy = 0
            _price_sell = 0
            _nb_trade = 0
            _start_buy = 0
            _start_sell = 0
            _cpt_buy = 0
            _cpt_sell = 0 
            _pnl = 0
            _win = 0
            _los = 0
            PNL = []
            print('Pour un TP de ',col.Fore.YELLOW,_tp,col.Style.RESET_ALL,'et un SL de ',col.Fore.YELLOW,_sl,col.Style.RESET_ALL)    
            for i in range(len(X)):
                
                # Brute Closing
                if i == len(X)-1 :
                    if _open_buy == 1:
                        print("fermeture long en l'air")
                        _pnl = _size * (X.Close[i] - _price_buy)
                        _nb_trade += 1
                        _open_buy = 999
                        _open_sell = 999
                        _price_buy = 0
                        PNL.append(_pnl)
                        if _pnl > 0:
                            _win += 1
                        else:
                            _los += 1

                    if _open_sell == 1:
                        print("fermeture short en l'air")
                        _pnl =  - _size * (X.Close[i] - _price_sell)
                        _nb_trade += 1
                        _open_sell = 999
                        _open_buy = 999
                        _price_sell = 0
                        PNL.append(_pnl)
                        if _pnl > 0:
                            _win += 1
                        else:
                            _los += 1

                # Open Long
                if X.Signal[i] == 1 and _open_buy == 0 and _open_sell == 0:
                    _price_buy = X.Close[i]
                    _open_buy = 1
                    _start_buy = i
                    _cpt_buy += 1

                # TP Close Short
                if (X.Close[i] - _price_sell)/_price_sell <= - _tp and _open_buy == 0 and _open_sell == 1:
                    _pnl =  - _size * (X.Close[i] - _price_sell)
                    _nb_trade += 1
                    _open_sell = 0
                    _price_sell = 0
                    PNL.append(_pnl)
                    if _pnl > 0:
                        _win += 1
                    else:
                        _los += 1
                
                # SL Close Short
                if (X.Close[i] - _price_sell)/_price_sell >=  _sl and _open_buy == 0 and _open_sell == 1:
                    _pnl =  - _size * (X.Close[i] - _price_sell)
                    _nb_trade += 1
                    _open_sell = 0
                    _price_sell = 0
                    PNL.append(_pnl)
                    if _pnl > 0:
                        _win += 1
                    else:
                        _los += 1

                # Open Short
                if X.Signal[i] == -1 and _open_buy == 0 and _open_sell == 0:
                    _price_sell = X.Close[i]
                    _open_sell = 1
                    _start_sell = i
                    _cpt_sell += 1

                # TP Close Long
                if (X.Close[i] - _price_buy)/_price_buy >=  _tp and _open_sell == 0 and _open_buy == 1:
                    _pnl = _size * (X.Close[i] - _price_buy)
                    _nb_trade += 1
                    _open_buy = 0
                    _price_buy = 0
                    PNL.append(_pnl)
                    if _pnl > 0:
                        _win += 1
                    else:
                        _los += 1

                # SL Close Short
                if (X.Close[i] - _price_buy)/_price_buy <= - _sl and _open_sell == 0 and _open_buy == 1:
                    _pnl = _size * (X.Close[i] - _price_buy)
                    _nb_trade += 1
                    _open_buy = 0
                    _price_buy = 0
                    PNL.append(_pnl)
                    if _pnl > 0:
                        _win += 1
                    else:
                        _los += 1

            try:
                print('Winner Ratio :',round((_win/(_win+_los))*100,2),'%')
            except:
                if _win == 0:
                    print('Aucun Winner')
                else:
                    print('Probleme avec le Ratio')
            try:
                print('Profit Factor :',col.Fore.BLUE,round(abs(sum(list(filter(lambda x: x > 0, PNL))) / sum(list(filter(lambda x: x <= 0, PNL)))),2),'%',col.Style.RESET_ALL)
            except:
                print(col.Fore.RED,'PNL à O',col.Style.RESET_ALL)
            
            print('Nombre de trades :',_nb_trade)
            print('Nombre de posistions non fermées',(_cpt_buy+_cpt_sell)-_nb_trade)
            print('Captital initial :',_capital)
            print('Taille des positions :',_size)
            if sum(PNL) > 0:
                print('Gain :',col.Fore.GREEN,sum(PNL),col.Style.RESET_ALL)
            else:
                print('Gain :',col.Fore.RED,sum(PNL),col.Style.RESET_ALL)
            print('Capital Final :',_capital + sum(PNL))
            print('\n________________________________________________________________________________________\n')
    _t2 = dt.datetime.now()
    print("Temps d'excution du module",str((_t2 - _t1)))
    return()

