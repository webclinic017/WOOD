#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

import datetime as dt 
import pprint

from quanTest.symbol    import SYMBOL
from quanTest.portfolio import PORTFOLIO 


def showPortfolioState(p, option = "") : 
    print ("============================")
    print ("PORTFOLIO STATE : ")
    print ("Balance          = ",p.balance) 
    print ("Equity           = ",p.equity)
    print ("Available margin = ",p.availableMargin) 
    print ("Used margin      = ",p.usedMargin)
    print ("Margin Level     = ",p.marginLevel) 
    
    if "positions" in option : 
        
        print ("OPEN POSITIONS : ")
        for pos in p.openPositions : 
            print ("-----------------------")
            pprint.pprint (pos.__dict__)
    
    if "pending orders" in option : 
        
        print ("PENDING ORDERS : ")
        for order in p.pendingOrders : 
            print ("-----------------------") 
            pprint.pprint(order.__dict__)
    
    if "executed orders" in option : 
        
        print ("EXECUTED ORDERS : ") 
        for order in p.executedOrders : 
            print ("-----------------------") 
            pprint.pprint(order.__dict__)

 



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
p.addSymbol(btcusd)


# We retrieve the actual values of our assets 
eurusd_price = 1.20103
eurusd_spread = 0.00005

btcusd_price = 52877.50 
btcusd_spread = 150.00

variation = 0.005



# We create orders
#option = "positions, pending orders, executed orders"
#option = ""
option = "pending orders, executed orders"

print ("#######################################")
print ("DAY 1") 
print ("#######################################")
# eurusd.setCurrentPrice(bidopen = eurusd_price - eurusd_spread, askopen = eurusd_price, time = dt.datetime.today())
# btcusd.setCurrentPrice(bidopen = btcusd_price - btcusd_spread, askopen = btcusd_price, time = dt.datetime.today())

p.symbols.get("EUR.USD").setCurrentPrice(bidopen = eurusd_price - eurusd_spread, askopen = eurusd_price, time = dt.datetime.today())
p.symbols.get("BTC.USD").setCurrentPrice(bidopen = btcusd_price - btcusd_spread, askopen = btcusd_price, time = dt.datetime.today())

# pprint.pprint(eurusd.__dict__)

showPortfolioState(p, option = option)
p.placeOrder("EUR.USD",
             action     = "long", 
             orderType  = "LMT", 
             volume     = 0.1, 
             stoploss   = 1.19, 
             takeprofit = 1.21, 
             lmtPrice   = 1.205)
# for pos in p.openPositions : 
#     for symbol in [eurusd, btcusd] : 
#         if symbol.symbolName == pos.symbol: 
#             p.updatePosition(symbol, pos)

# for order in p.pendingOrders : 
#     p.checkPendingOrder(p.symbols.get("EUR.USD"), order) 

# p.updatePortfolio()
# for order in p.pendingOrders : 
#     p.checkPendingOrder(p.symbols.get("EUR.USD"), order) 
p.update()
showPortfolioState(p, option = option)

print ("#######################################")
print ("DAY 2") 
print ("#######################################")
# eurusd.setCurrentPrice(bidopen = (eurusd_price - eurusd_spread)*(1. + variation), 
#                        askopen = (eurusd_price)*(1. + variation), time = dt.datetime.today())
# btcusd.setCurrentPrice(bidopen = (btcusd_price - btcusd_spread)*(1 + variation), 
#                        askopen = (btcusd_price)*(1. + variation), time = dt.datetime.today())

p.symbols.get("EUR.USD").setCurrentPrice(bidopen = (eurusd_price - eurusd_spread)*(1. + variation), 
                       askopen = (eurusd_price)*(1. + variation), time = dt.datetime.today())
p.symbols.get("BTC.USD").setCurrentPrice(bidopen = (btcusd_price - btcusd_spread)*(1 + variation), 
                       askopen = (btcusd_price)*(1. + variation), time = dt.datetime.today())

# pprint.pprint(eurusd.__dict__)

# for order in p.pendingOrders : 
#     p.checkPendingOrder(eurusd, order) 

# # for pos in p.openPositions : 
# #     for symbol in [eurusd, btcusd] : 
# #         if symbol.symbolName == pos.symbol: 
# #             p.updatePosition(symbol, pos)
# p.updatePortfolio()
p.update()
showPortfolioState(p, option = option)

print ("#######################################")
print ("DAY 3") 
print ("#######################################")
# eurusd.setCurrentPrice(bidopen = (eurusd_price - eurusd_spread)*(1. + 2*variation), 
#                        askopen = (eurusd_price)*(1. + 2*variation), time = dt.datetime.today())
# btcusd.setCurrentPrice(bidopen = (btcusd_price - btcusd_spread)*(1 + 2*variation), 
#                        askopen = (btcusd_price)*(1. + 2*variation), time = dt.datetime.today())

p.symbols.get("EUR.USD").setCurrentPrice(bidopen = (eurusd_price - eurusd_spread)*(1. + 2*variation), 
                       askopen = (eurusd_price)*(1. + 2*variation), time = dt.datetime.today())
p.symbols.get("BTC.USD").setCurrentPrice(bidopen = (btcusd_price - btcusd_spread)*(1 + 2*variation), 
                       askopen = (btcusd_price)*(1. + 2*variation), time = dt.datetime.today())
# pprint.pprint(eurusd.__dict__)




# for order in p.pendingOrders : 
#     p.checkPendingOrder(eurusd, order) 

# # p.closePosition(eurusd, p.openPositions[0])
# p.updatePortfolio()
p.update()
showPortfolioState(p, option = option)

# p.placeOrder(btcusd, 
#              action     = "short", 
#              orderType  = "MKT", 
#              volume     = 5, 
#              stoploss   = 0., 
#              takeprofit = 10000000.)
# showPortfolioState(p, option = option)
# p.closePosition(btcusd, p.openPositions[1].orderID)
# showPortfolioState(p, option = option)












