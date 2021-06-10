#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

import numpy as np 
import pandas as pd 
import datetime as dt 
import matplotlib.pyplot as plt 
import pprint

from quanTest.symbol    import SYMBOL
from quanTest.portfolio import PORTFOLIO  


###############################################################################
# Some important functions  
###############################################################################
def generatePriceValues(numberCandles, 
                        priceIni, 
                        priceEnd, 
                        dateIni, 
                        dateEnd,
                        sigma  = 0.01, 
                        subSigma1 = 0.01, 
                        subSigma2 = 0.01,
                        spreadMean = 0.001,
                        spreadVar  = 0.0001) : 
    
    # We generate a convenient global price array
    locPriceIni = 0. 
    diffusionPrice = list([0.])
    globalPrice_ = list([locPriceIni])
    for i in range(1, numberCandles) : 
        variation  = np.random.normal(loc = 0., scale = sigma)
        diffusionPrice.append(variation)
        globalPrice_.append(globalPrice_[-1] + diffusionPrice[-1])
    globalPrice_ = abs(priceIni + np.array(globalPrice_)/(globalPrice_[-1] - globalPrice_[0])*(priceEnd - priceIni))
    
    # We generate a datetime array 
    dateList = np.linspace(pd.Timestamp(str(dateIni)).value, pd.Timestamp(str(dateEnd)).value, num = numberCandles)
    dateList = pd.to_datetime(dateList)
    
    # We generate the standard OHLC bars 
    openPrice = list() 
    highPrice = list() 
    lowPrice  = list() 
    closePrice= list() 
    for i in range(numberCandles) : 
        variation1  = np.random.normal(loc = 0., scale = subSigma1)
        variation2  = np.random.normal(loc = 0., scale = subSigma2)
        openPrice.append(globalPrice_[i] + variation1) 
        closePrice.append(globalPrice_[i] + variation2) 
        highPrice.append(max(openPrice[-1], closePrice[-1]) + abs(variation1))
        lowPrice.append(min(openPrice[-1], closePrice[-1]) - abs(variation2))
        
    # We generate data containing the spread variations 
    askOpen = list() 
    askHigh = list() 
    askLow  = list() 
    askClose= list() 
    bidOpen = list() 
    bidHigh = list() 
    bidLow  = list() 
    bidClose= list()
    for i in range(numberCandles) : 
        spread = abs(np.random.normal(loc = spreadMean, scale = spreadVar)) 
        
        askOpen.append(openPrice[i])
        askHigh.append(highPrice[i])
        askLow.append(lowPrice[i])
        askClose.append(closePrice[i])
        
        bidOpen.append(openPrice[i] + spread)
        bidHigh.append(highPrice[i] + spread)
        bidLow.append(lowPrice[i] + spread)
        bidClose.append(closePrice[i] + spread)
    
    # We put all the data in a dataframe 
    d = {"askopen" : askOpen, 
         "askhigh" : askHigh, 
         "asklow"  : askLow, 
         "askclose": askClose, 
         "bidopen" : bidOpen, 
         "bidhigh" : bidHigh, 
         "bidlow"  : bidLow, 
         "bidclose": bidClose, 
         "time"    : list(dateList)}
    
    df = pd.DataFrame(data = d) 
    return df 
    # print (df)
    
    
    # print("Price ini :", globalPrice_[0], ", price end : ",globalPrice_[-1], " len : ", len(globalPrice_))
    # plt.plot(dateList, globalPrice_)
    
    # plt.plot(dateList, openPrice, c = "blue", ls = '-')
    # plt.plot(dateList, highPrice, c = "blue", ls = '--')
    # plt.plot(dateList, lowPrice, c = "red", ls = '--')
    # plt.plot(dateList, closePrice, c = "red", ls = '-')


###############################################################################
# Simulation initialisation   
###############################################################################
# We first design our historical data 
data_eurusd = generatePriceValues(100000, 1.10, 1.20, dt.datetime(2021, 1, 12, 12, 12), dt.datetime(2021, 4, 12, 12, 12))

# We define an asset
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

# We initialize our portfolio 
p = PORTFOLIO(initialDeposit                  = 100000,                # The initial client deposit 
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




inPosition = False
for i in range(len(data_eurusd)) : 
    print ("Loop n°",i,"/",len(data_eurusd))
    p.symbols.get("EUR.USD").setCurrentPrice(bidopen = data_eurusd["bidopen"].iloc[i], 
                                             askopen = data_eurusd["askopen"].iloc[i], 
                                             time    = data_eurusd["time"].iloc[i])
    if not inPosition : 
        stoploss   = p.symbols.get("EUR.USD").bidopen*(1 - 0.05) 
        takeprofit = p.symbols.get("EUR.USD").bidopen*(1 + 0.1) 
        # p.placeOrder("EUR.USD",
        #              action     = "long", 
        #              orderType  = "MKT", 
        #              volume     = 0.1, 
        #              stoploss   = stoploss, 
        #              takeprofit = takeprofit, 
        #              lmtPrice   = None)
        p.placeOrder("EUR.USD",
                     action     = "short", 
                     orderType  = "MKT", 
                     volume     = 0.1, 
                     stoploss   = takeprofit, 
                     takeprofit = stoploss, 
                     lmtPrice   = None)
    
    if len(p.openPositions) > 0 : 
        inPosition = True 
    else : 
        inPosition = False 
        
        
    p.update()

















