##### VANILLA BT SUR SIGNALS.shift(-1) déjà shifté ==> objectif Vérifier la robustesse de l'output

def retrace(X):
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
        print('Winner Ratio :',round((1-(_los/_win))*100,2),'%')
    except:
        if _win == 0:
            print('Aucun Winner')
        else:
            print('Probleme avec le Ratio')
    try:
        print('Profit Ratio :',round(abs(sum(list(filter(lambda x: x > 0, PNL))) / sum(list(filter(lambda x: x <= 0, PNL)))),2),'%')
    except:
        print('PNL à O')
    
    print('Nombre de trades :',_nb_trade)
    print('Nombre de posistions non fermée',(_cpt_buy+_cpt_sell)-_nb_trade)
    print('Captital initial :',_capital)
    print('Taille des positions :',_size)
    print('Gain :',sum(PNL))
    print('Capital Final :',_capital + sum(PNL))
    return()