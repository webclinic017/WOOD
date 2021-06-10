#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

import datetime as dt 
import pprint

from quanTest.symbol     import SYMBOL
from quanTest.portfolio  import PORTFOLIO 
from quanTest.data       import PRICE 
from quanTest.data       import PRICE_TABLE
from quanTest.simulation import SIMULATION

""" 
===============================================================================
INITIALIZATION STEP
===============================================================================
"""

# We initialize some assets 
eurusd = SYMBOL(symbolName              = "EUR.USD",
                contractSize            = 100000, 
                marginCurrency          = "USD", # Can be any existing currency 
                profitCalculationMethod = "Forex", # "CFD", "Forex", "Stock", "CFD-Index"
                marginRequestMethod     = "Forex", # "CFD", "Forex", "Stock", "CFD-Index"
                marginPercentage        = 100, 
                execution               = "Market", 
                minimalVolume           = 0.01, 
                maximalVolume           = 100.0, 
                volumeStep              = 0.01, 
                precision               = 5,        # Price precision (3 means 1 point = 0.001)
                exchangeType            = "Point", # "Point", "Percentage"
                exchangeLong            = 6.88, 
                exchangeShort           = 0.63)

btcusd = SYMBOL(symbolName              = "BTC.USD",
                contractSize            = 1, 
                marginCurrency          = "USD", # Can be any existing currency 
                profitCalculationMethod = "CFD", # "CFD", "Forex", "Stock", "CFD-Index"
                marginRequestMethod     = "CFD", # "CFD", "Forex", "Stock", "CFD-Index"
                marginPercentage        = 100, 
                execution               = "Market", 
                minimalVolume           = 0.10, 
                maximalVolume           = 5.00, 
                volumeStep              = 0.10, 
                precision               = 2,        # Price precision (3 means 1 point = 0.001)
                exchangeType            = "Percentage", # "Point", "Percentage"
                exchangeLong            = -30, 
                exchangeShort           = 10)


# We initialize our portfolio 
p = PORTFOLIO(initialDeposit                  = 10000000,                # The initial client deposit 
              leverage                        = 30,                    # The leverage value (margin = initialDeposit*leverage)
              currency                        = "USD",                # The currency 
              positions                       = "long & short",       # "long", "short" or "long & short"
              marginCallTreeshold             = 100,                  # If marginLevel < marginCallTreeshold : Warning (no more trading allowed)
              marginMinimum                   = 50,                   # If marginLevel < marginMinimum : Automatically close all losing positions 
              minimumBalance                  = 200,                  # If balance < minimumBalance : No more trading allowed 
              maximumProfit                   = 10000,                # If balance - inialDeposit > maximumProfit : No more trading allowed 
              maximumDrawDown                 = 70,                   # If drawDown < maximumDrawDown : No more trading allowed 
              maximumConsecutiveLoss          = 5000,                 # If valueLossSerie > maximumConsecutiveLoss : No more trading allowed 
              maximumConsecutiveGain          = 10000,                # If valueGainSerie > maximumConsecutiveGain : No more trading allowed 
              maximumNumberOfConsecutiveGains = 30)


p.addSymbol(eurusd)
p.addSymbol(btcusd)






path  = "/home/loann/Travail/Quantums/Travaux/Données/Finnhub/Stocks/"
path += "GOOG_112019-102020_M1.csv"

price1 = PRICE("EUR.USD") 


price1.setColumnsTitle(askOpen    ="askopen", 
                      askHigh     ="askhigh",
                      askLow      ="asklow",
                      askClose    ="askclose", 
                      bidOpen     ="askopen",
                      bidHigh     ="askhigh",
                      bidLow      ="asklow",
                      bidClose    ="askclose",
                      date        ="date",
                      dateFormat  = "%Y-%m-%d %H:%M:%S", 
                      volume      ="volume")

price1.read(path)
price1.setBaseTimeframe(timeframe = dt.timedelta(minutes = 1))
price1.fillMissingData()
price1.shiftMarketTime(timeshift = 0)

price1.dataTimeZone   = 0
price1.marketTimeZone = 0 

price1.marketOpeningHour = "09:30"
price1.marketClosingHour = "17:30"
price1.marketLunch = "12:30-13:30"

price1.setMarketState()


path  = "/home/loann/Travail/Quantums/Travaux/Données/Finnhub/Stocks/"
path += "AAPL_112019-102020_M1.csv"

price2 = PRICE("BTC.USD") 


price2.setColumnsTitle(askOpen    ="askopen", 
                      askHigh    ="askhigh",
                      askLow     ="asklow",
                      askClose   ="askclose", 
                      bidOpen    ="askopen",
                      bidHigh    ="askhigh",
                      bidLow     ="asklow",
                      bidClose   ="askclose",
                      date       ="date",
                      dateFormat = "%Y-%m-%d %H:%M:%S", 
                      volume     ="volume")

price2.read(path)
price2.setBaseTimeframe(timeframe = dt.timedelta(minutes = 1))
price2.fillMissingData()
price2.shiftMarketTime(timeshift = 0)

price2.dataTimeZone   = 0
price2.marketTimeZone = 0

price2.marketOpeningHour = "09:30"
price2.marketClosingHour = "17:30"
price2.marketLunch = "12:30-13:30"

price2.setMarketState()

table = PRICE_TABLE([price1, price2]) 
table.synchronize()



""" 
===============================================================================
SIMULATION STEP
===============================================================================
"""
sim = SIMULATION(p, table)

sim.startIndex = 20
sim.stopIndex = 40000 


sim.strategyPath = "/home/loann/Bureau/Dossiers de travail/Quantums_Framework/BackTestModule/tests/"
sim.strategyFile = "exampleStrategy"

sim.importStrategy()
sim.parametersCheck()
sim.run()








