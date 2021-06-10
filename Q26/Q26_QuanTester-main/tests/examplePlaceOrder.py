import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"..")
sys.path.append(filename)

from pprint import pprint 
import datetime as dt 

from quanTest.symbol    import SYMBOL
from quanTest.portfolio import PORTFOLIO 


eurusd = SYMBOL(symbolName              = "EUR.USD",
                contractSize            = 100000, 
                marginCurrency          = "EUR", # Can be any existing currency 
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

eurusd.setCurrentPrice(bidopen = 1.20103, askopen = 1.20095, time = dt.datetime.today())

p = PORTFOLIO(initialDeposit                  = 100000,                # The initial client deposit 
              leverage                        = 30,                    # The leverage value (margin = initialDeposit*leverage)
              currency                        = "USD",                # The currency 
              positions                       = "Long & Short",       # "Long", "Short" or "Long & Short"
              marginCallTreeshold             = 100,                  # If marginLevel < marginCallTreeshold : Warning (no more trading allowed)
              marginMinimum                   = 50,                   # If marginLevel < marginMinimum : Automatically close all losing positions 
              minimumBalance                  = 200,                  # If balance < minimumBalance : No more trading allowed 
              maximumProfit                   = 10000,                # If balance - inialDeposit > maximumProfit : No more trading allowed 
              maximumDrawDown                 = 70,                   # If drawDown < maximumDrawDown : No more trading allowed 
              maximumConsecutiveLoss          = 5000,                 # If valueLossSerie > maximumConsecutiveLoss : No more trading allowed 
              maximumConsecutiveGain          = 10000,                # If valueGainSerie > maximumConsecutiveGain : No more trading allowed 
              maximumNumberOfConsecutiveGains = 30)

p.createOrder(action     = "short", 
             orderType  = "bracket", 
             volume     = 0.1, 
             stoploss   = 1.21, 
             takeprofit = 1.19)

#p.createOrder(action     = "long", 
#             orderType  = "bracket", 
#             volume     = 0.1, 
#             stoploss   = 1.19, 
#             takeprofit = 1.21)


print ("===================================")
print ("INITIAL PORTFOLIO STATE")
print ("Balance           = ",p.balance) 
print ("Available margin  = ",p.availableMargin)
print ("Margin level      = ",p.marginLevel) 


print ("CREATED ORDERS")
for i in range(len(p.pendingOrders)) : 
    print ("Pending : ",p.pendingOrders[i].__dict__)
print("===============================================")

p.executeOrder(eurusd, 
               p.pendingOrders[0])

print ("AFTER EXECUTION")
for i in range(len(p.pendingOrders)) : 
    print ("Pending : ",p.pendingOrders[i].__dict__)
for i in range(len(p.executedOrders)) : 
    print ("Executed : ",p.executedOrders[i].__dict__) 
for i in range(len(p.openPositions)) : 
    print ("Open Order : ",p.openPositions[i].__dict__)
print ("===================================")
print ("INTERMEDIATE PORTFOLIO STATE")
print ("Balance           = ",p.balance) 
print ("Available margin  = ",p.availableMargin)
print ("Margin level      = ",p.marginLevel) 
print("===============================================")

eurusd.setCurrentPrice(bidopen = 1.19103, askopen = 1.19095, time = dt.datetime.today())
p.updatePosition(eurusd, p.openPositions[0]) 
for i in range(len(p.openPositions)) : 
    print ("Open Order : ",p.openPositions[i].__dict__)
print ("INTERMEDIATE PORTFOLIO STATE 2")
print ("Balance           = ",p.balance) 
print ("Available margin  = ",p.availableMargin)
print ("Margin level      = ",p.marginLevel) 
print("===============================================")


eurusd.setCurrentPrice(bidopen = 1.18103, askopen = 1.18095, time = dt.datetime.today())
#eurusd.setCurrentPrice(bidopen = 1.11103, askopen = 1.11095, time = dt.datetime.today())
p.updatePosition(eurusd, p.openPositions[0]) 
p.updatePortfolio()
for i in range(len(p.openPositions)) : 
    print ("Open Order : ",p.openPositions[i].__dict__)
print ("INTERMEDIATE PORTFOLIO STATE 3")
print ("Balance           = ",p.balance) 
print ("Available margin  = ",p.availableMargin)
print ("Margin level      = ",p.marginLevel) 
print("===============================================")

for i in range(len(p.pendingOrders)) : 
    try : 
        p.checkPendingOrder(eurusd, 
                            p.pendingOrders[i]) 
    except : 
        pass 

print ("AFTER CHECK")
for i in range(len(p.pendingOrders)) : 
    print ("Pending : ",p.pendingOrders[i].__dict__)
for i in range(len(p.executedOrders)) : 
    print ("Executed : ",p.executedOrders[i].__dict__) 
for i in range(len(p.openPositions)) : 
    print ("Open Positions : ",p.openPositions[i].__dict__)
for i in range(len(p.closedPositions)) : 
    print ("Closed Positions : ",p.closedPositions[i].__dict__)
print("===============================================")
print ("END PORTFOLIO STATE")
print ("Balance           = ",p.balance) 
print ("Available margin  = ",p.availableMargin)
print ("Margin level      = ",p.marginLevel) 
