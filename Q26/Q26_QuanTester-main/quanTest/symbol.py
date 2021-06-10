#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
FUNCTIONNALITIES TO ADD : 
    - The possibility to cut the symbols array if we want to specify the time range over which we want to simulate. 

"""

class SYMBOL : 
    def __init__(
        self, 
        symbolName              = "GOLD.USD",
        dataTableName           = None,
        contractSize            = 100, 
        marginCurrency          = "USD", # Can be any existing currency 
        profitCalculationMethod = "CFD", # "CFD", "Forex", "Stock", "CFD-Index"
        marginRequestMethod     = "CFD", # "CFD", "Forex", "Stock", "CFD-Index"
        marginPercentage        = 100, 
        execution               = "Market", 
        minimalVolume           = 0.01, 
        maximalVolume           = 25.0, 
        volumeStep              = 0.01, 
        precision               = 3,        # Price precision (3 means 1 point = 0.001)
        exchangeType            = "Point", # "Point", "Percentage"
        exchangeLong            = 99.5, 
        exchangeShort           = 58.2 
    ) : 
        # These values are static over all the backtest 
        self.symbolName              = symbolName
        self.dataTableName           = dataTableName
        self.contractSize            = contractSize 
        self.marginCurrency          = marginCurrency 
        self.profitCalculationMethod = profitCalculationMethod 
        self.marginRequestMethod     = marginRequestMethod 
        self.marginPercentage        = marginPercentage 
        self.execution               = execution 
        self.minimalVolume           = minimalVolume 
        self.maximalVolume           = maximalVolume 
        self.volumeStep              = volumeStep
        self.precision               = precision
        self.exchangeType            = exchangeType 

        # These values are the brokerage fees (swap) and can evolve with time 
        self.exchangeLong            = exchangeLong 
        self.exchangeShort           = exchangeShort
        
        # These values are going to be updated at each new price value. 
        self.askopen                 = None 
        self.askhigh                 = None 
        self.asklow                  = None 
        self.askclose                = None 
        self.askprice                = None # Price to be used as execution price 
        self.bidopen                 = None 
        self.bidhigh                 = None 
        self.bidlow                  = None 
        self.bidclose                = None 
        self.bidprice                = None # Price to be used as execution price 
        self.volume                  = None 
        self.time                    = None 
        self.marketState             = "open" 

        return 
    
    def setCurrentMarketState(
        self, 
        state
    ) : 
        """ 
        Define the current market state : 
            - "open"      : Buy and Sell orders are both allowed 
            - "closed"    : No trading allowed 
            - "buy only"  : Only buy orders allowed 
            - "sell only" : Only sell orders allowed
        """
        self.marketState = state
    
    def setCurrentPrice(
        self, 
        askopen  = None, 
        askhigh  = None, 
        asklow   = None, 
        askclose = None, 
        bidopen  = None, 
        bidhigh  = None, 
        bidlow   = None, 
        bidclose = None,
        volume   = None, 
        time     = None, 
        askprice = None, 
        bidprice = None  
    ) : 
        if askopen is not None : 
            self.askopen = askopen
        if askhigh is not None : 
            self.askhigh = askhigh
        if asklow is not None : 
            self.asklow = asklow 
        if askclose is not None : 
            self.askclose = askclose 
        if bidopen is not None : 
            self.bidopen = bidopen
        if bidhigh is not None : 
            self.bidhigh = bidhigh
        if bidlow is not None : 
            self.bidlow = bidlow 
        if bidclose is not None : 
            self.bidclose = bidclose 
        if volume is not None : 
            self.volume = volume 
        if time is not None : 
            self.time   = time 
        if askprice is not None : 
            self.askprice = askprice 
        if bidprice is not None : 
            self.bidprice = bidprice 

"""   
class SYMBOL_TABLE : 

    def __init__(self, symbolList) : 

        self.symbolDict = dict()
        self.size       = None 
        for symbol in symbolList : 
            self.symbolDict.update({symbol.symbolName : symbol})
            self.size = len(symbol.time)

        
        return 
    
    def symbol(self, symbolName) : 
         
        #Function that allows to access to symbol informations but also to 
        #edit them by reference. Example : 
        #    self.symbol(symbolName).attr = newAttrValue 
        
        return self.symbolDict.get(symbolName)
"""
    
