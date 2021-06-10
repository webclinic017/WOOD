#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:07:54 2021

@author: loann
"""
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

import numpy as np 
import pandas as pd 
import datetime as dt 
import matplotlib.pyplot as plt 
import pprint
import copy 

from quanTest.symbol     import SYMBOL
from quanTest.portfolio  import PORTFOLIO 
from quanTest.data import PRICE 
from quanTest.data import PRICE_TABLE
from quanTest.simulation import SIMULATION

""" 
===============================================================================
INITIALIZATION STEP
===============================================================================
"""

path  = "/home/loann/Travail/Quantums/Travaux/Données/Other/HDD/"
path += "EURUSD_m5_BidAndAsk.csv"

price = PRICE("EUR.USD") 

price.setColumnsTitle(askOpen        ="OpenAsk", 
                      askHigh        ="HighAsk",
                      askLow         ="LowAsk",
                      askClose       ="CloseAsk", 
                      bidOpen        ="OpenBid",
                      bidHigh        ="HighBid",
                      bidLow         ="LowBid",
                      bidClose       ="CloseBid",
                      dateFormat     ="%m-%d-%Y %H:%M:%S", 
                      volume         ="Total Ticks",
                      splitDaysHours =True, 
                      days           ="Date", 
                      hours          ="Time")

price.read(path)
price.setBaseTimeframe(timeframe = dt.timedelta(minutes = 5))
price.fillMissingData()

price.shiftMarketTime(timeshift = 0)
price.dataTimeZone   = 0
price.marketTimeZone = 0
price.marketOpeningHour = "00:00"
price.marketClosingHour = "24:00"
price.marketLunch = None
price.daysOfWeek = [0, 1, 2, 3, 4]

price.setMarketState() 

price_H1 = copy.deepcopy(price)
price_H1.resampleData("01:00", name = "EUR.USD_H1")

table = PRICE_TABLE([price, price_H1]) 
table.synchronize()

symbol = SYMBOL(symbolName              = "EUR.USD",
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

# We initialize our portfolio 
p = PORTFOLIO(initialDeposit                  = 100000,                # The initial client deposit 
              leverage                        = 10,                    # The leverage value (margin = initialDeposit*leverage)
              currency                        = "USD",                # The currency 
              positions                       = "long & short",       # "long", "short" or "long & short"
              marginCallTreeshold             = 100,                  # If marginLevel < marginCallTreeshold : Warning (no more trading allowed)
              marginMinimum                   = 50,                   # If marginLevel < marginMinimum : Automatically close all losing positions 
              minimumBalance                  = 50000,                  # If balance < minimumBalance : No more trading allowed 
              maximumProfit                   = 100000,                # If balance - inialDeposit > maximumProfit : No more trading allowed 
              maximumDrawDown                 = 70,                   # If drawDown < maximumDrawDown : No more trading allowed 
              maximumConsecutiveLoss          = 50000,                 # If valueLossSerie > maximumConsecutiveLoss : No more trading allowed 
              maximumConsecutiveGain          = 50000,                # If valueGainSerie > maximumConsecutiveGain : No more trading allowed 
              maximumNumberOfConsecutiveGains = 30)


p.addSymbol(symbol)

""" 
===============================================================================
SIMULATION STEP
===============================================================================
"""
sim = SIMULATION(p, table)

sim.subLoopModel = "close only"
sim.maxHstDataSize = 2000
sim.startIndex = 3000
#sim.stopIndex  = 3010
sim.logEvery = 1000


# sim.strategyPath = "/home/loann/Travail/Quantums/Travaux/Algorithmes/Quantums_Framework/Q26_StratPool/strategies/Examples/"
# sim.strategyFile = "simpleExample"
sim.strategyPath = "/home/loann/Travail/Quantums/Travaux/Algorithmes/Quantums_Framework/Q26_StratPool/strategies/RaptorTrend/"
sim.strategyFile = "raptorTrend"

sim.importStrategy()
sim.parametersCheck()
sim.run()

