#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import datetime as dt 

import quanTest.financialTools as financialTools
from quanTest.order import ORDER
from quanTest.position import POSITION


class PORTFOLIO : 

    def __init__(self, 
                 initialDeposit                  = 10000,                # The initial client deposit 
                 leverage                        = 1,                    # The leverage value (margin = initialDeposit*leverage)
                 currency                        = "USD",                # The currency 
                 positions                       = "long & short",       # "long", "short" or "long & short"
                 marginCallTreeshold             = 100,                  # If marginLevel < marginCallTreeshold : Warning (no more trading allowed)
                 marginMinimum                   = 50,                   # If marginLevel < marginMinimum : Automatically close all losing positions 
                 minimumBalance                  = 200,                  # If balance < minimumBalance : No more trading allowed 
                 maximumProfit                   = 10000,                # If balance - inialDeposit > maximumProfit : No more trading allowed 
                 maximumDrawDown                 = 70,                   # If drawDown < maximumDrawDown : No more trading allowed 
                 maximumConsecutiveLoss          = 5000,                 # If valueLossSerie > maximumConsecutiveLoss : No more trading allowed 
                 maximumConsecutiveGain          = 10000,                # If valueGainSerie > maximumConsecutiveGain : No more trading allowed 
                 maximumNumberOfConsecutiveGains = 30) :                 # If numberLossSerie > maximumNumberOfConsecutiveGains : No more trading allowed 
            
        # Initial parameters (static)
        self.initialDeposit                         = initialDeposit 
        self.initialAvailableMargin                 = initialDeposit
        self.leverage                               = leverage 
        self.currency                               = currency 

        # Constraint parameters (static, but can evolve if needed)
        self.positions                              = positions 
        self.marginCallTreeshold                    = marginCallTreeshold
        self.marginMinimum                          = marginMinimum 
        self.minimumBalance                         = minimumBalance 
        self.maximumProfit                          = maximumProfit 
        self.maximumDrawDown                        = maximumDrawDown
        self.maximumConsecutiveLoss                 = maximumConsecutiveLoss 
        self.maximumConsecutiveGain                 = maximumConsecutiveGain 
        self.maximumNumberOfConsecutiveGains        = maximumNumberOfConsecutiveGains

        # Simulation parameters (dynamic) 
        self.balance                                = self.initialDeposit 
        self.availableMargin                        = self.initialAvailableMargin 
        self.usedMargin                             = 0. 
        self.equity                                 = self.initialAvailableMargin
        self.marginLevel                            = np.inf 
        self.openPositions                          = list()
        self.closedPositions                        = list()
        self.pendingOrders                          = list() 
        self.executedOrders                         = list() 
        self.equityCurve                            = list([initialDeposit])
        self.currentValueLossSerie                  = 0.
        self.currentValueGainSerie                  = 0. 
        self.currentDrawDown                        = 0. 
        self.currentMaximumNumberOfConsecutiveGains = 0 

        # Symbol objects  
        self.symbols                                = dict() 

        # Historical data price 
        self.historicalDataPrice                    = None 
        self.historicalDataTimeframe                = None # [int] in minutes 

        # Trading authorisation 
        self.tradeAuthorisation                     = True 
        
        # Debug attributes 
        self.verbose                                = True 
        
        return 
    

    
    # Functions to be used at an higher level 
    def addSymbol(self, symbol) : 
        self.symbols.update({symbol.symbolName : symbol})

    def placeOrder(self, 
                   symbolName,
                   action     = "long",      # "long" or "short"
                   orderType  = "MKT",       # Order kind : "MKT", "MIT", "LMT"
                   volume     = 0.1,         # Volume 
                   stoploss   = None,          
                   takeprofit = None, 
                   lmtPrice   = None, 
                   auxPrice   = None ) : 
        
        symbol = self.symbols.get(symbolName)

        #print(symbol.__dict__)

        if symbol is not None and self.tradeAuthorisation : 

            orderList = self.createOrder(symbolName = symbolName, 
                                         action     = action,          # "long" or "short"
                                         orderType  = orderType,       # Order kind : "MKT", "MIT", "LMT"
                                         volume     = volume,          # Volume 
                                         stoploss   = stoploss,          
                                         takeprofit = takeprofit, 
                                         lmtPrice   = lmtPrice, 
                                         auxPrice   = auxPrice)

            self.checkPendingOrder(symbol, 
                                   orderList[0])
            
            return orderList 
        
        else : 
            
            return [False, False, False]
    
    def editSLOrder(self, 
                    symbolName, 
                    order, 
                    stoploss = None) : 
        if stoploss is not None : 
            orderID  = order.orderID 
            parentID = order.parentID
            if parentID is not None : 
                pendingOrderIndex = None 
                for i in range(len(self.pendingOrders)) : 
                    if (self.pendingOrders[i].orderID == orderID) : 
                        pendingOrderIndex = i 
                if pendingOrderIndex is not None : 
                    self.pendingOrders[pendingOrderIndex].lmtPrice = stoploss 

    def editTPOrder(self, 
                    symbolName, 
                    order, 
                    takeprofit = None) : 
        if takeprofit is not None : 
            orderID  = order.orderID 
            parentID = order.parentID
            if parentID is not None : 
                pendingOrderIndex = None 
                for i in range(len(self.pendingOrders)) : 
                    if (self.pendingOrders[i].orderID == orderID) : 
                        pendingOrderIndex = i 
                if pendingOrderIndex is not None : 
                    self.pendingOrders[pendingOrderIndex].lmtPrice = takeprofit 
    
    def cancelOrder(self, 
                    symbolName, 
                    order) : 

        inPendingOrders = False 
        for o in self.pendingOrders : 
            if order.orderID == o.orderID : 
                inPendingOrders = True 
        
        if inPendingOrders : 
        
            # Case where the order is a parent one 
            if order.parentID is None : 

                stillAnOrder = True 
                while (stillAnOrder) : 
                    pendingOrderIndex = None
                    for i in range(len(self.pendingOrders)) : 
                        if self.pendingOrders[i].orderID == order.orderID or self.pendingOrders[i].parentID == order.orderID : 
                            pendingOrderIndex = i 
                    if pendingOrderIndex is not None : 
                        self.executedOrders.append(self.pendingOrders[pendingOrderIndex])
                        del self.pendingOrders[pendingOrderIndex]
                    else : 
                        stillAnOrder = False 



            else : 
                stillAnOrder = True 
                while (stillAnOrder) : 
                    pendingOrderIndex = None
                    for i in range(len(self.pendingOrders)) : 
                        if self.pendingOrders[i].orderID == order.orderID or self.pendingOrders[i].parentID == order.orderID : 
                            pendingOrderIndex = i 
                    if pendingOrderIndex is not None : 
                        self.executedOrders.append(self.pendingOrders[pendingOrderIndex])
                        del self.pendingOrders[pendingOrderIndex]
                    else : 
                        stillAnOrder = False 

        
                                         
    
    def closePosition(self,
                      symbolName, 
                      order) : 
        
        symbol = self.symbols.get(symbolName)

        if symbol is not None : 

            orderID = order.orderID
            # We retrive the position in the portfolio 
            openPositionIndex = None 
            for i in range(len(self.openPositions)) : 
                if self.openPositions[i].orderID == orderID : 
                    openPositionIndex = i 
            
            if openPositionIndex is None : 
                print ("Error when retrieving the position")
            else : 
                self.updatePosition(symbol, self.openPositions[openPositionIndex])
                self.updatePortfolio()
                position = self.openPositions[openPositionIndex]
                

                # We create an order to close the current position 
                orderS           = ORDER()
                orderS.symbolName= symbolName
                orderS.parentID  = position.orderID 
                orderS.action    = "short" if position.action == "long" else "long" 
                orderS.orderType = "MKT" 
                orderS.volume    = position.volume 

                # We execute the order 
                self.executeOrder(symbol, 
                                orderS, 
                                type = "close") 
    
    def getActivePositions(self, 
                           symbolName) : 

        activePositions = list()
        for pos in self.openPositions : 
            if pos.symbol == symbolName : 
                activePositions.append(pos)
        
        return activePositions

    ###################################################################
    # Security check functions. 
    ###################################################################
    def checkExecuteOrder(self, 
                          symbol, 
                          order) : 

        executeOrder = True 
        # A. We check if we have the global trading authorisation 
        if not self.tradeAuthorisation : 
            if self.verbose : print ("Order cannot be executed. Trading un-authorised")
            executeOrder = False 
        # B. We check if the order action is authorised by the portfolio 
        if not order.action in self.positions : 
            if self.verbose : print ("Order action ",order.action," is not allowed by the portfolio parameters")
            executeOrder = False 

        # 1. We check if we have the request margin 
        if order.requestMargin > self.availableMargin : 
            if self.verbose : print ("Order cannot be executed. The request margin is higher than the available margin")
            executeOrder = False 
        if self.marginLevel < symbol.marginPercentage : 
            if self.verbose : print ("Order cannot be executed. Available margin lower than the limit margin")
            executeOrder = False 
        # 2. We check if the volume responds to the constraints 
        if order.volume > symbol.maximalVolume or order.volume < symbol.minimalVolume : 
            if self.verbose : print ("Order cannot be executed. The volume amount is not right.") 
            executeOrder = False 
        # 3. We check if the volume step responds to the constrains 
        if order.volume / symbol.volumeStep != int(order.volume / symbol.volumeStep) : 
            if self.verbose : print ("Order cannot be executed. The volume step is not right")
            executeOrder = False
        # 4. We check if the market is open or not 
        if symbol.marketState == "closed" : 
            if self.verbose : print ("Order cannot be executed. The market is closed")
            executeOrder = False 
        if symbol.marketState == "sell only" and order.action == "long" : 
            if self.verbose : print ("Order cannot be executed. Only sell orders are allowed")
            executeOrder = False 
        if symbol.marketState == "buy only" and order.action == "short" : 
            if self.verbose : print ("Order cannot be executed. Only buy orders are allowed")
            executeOrder = False 
        
        return executeOrder 
    

    def tradingAuthorisation(self) : 
        
        locAuthorisation = True 

        # 1. The available margin is lower than the margin call treeshold 
        if self.marginLevel < self.marginCallTreeshold : 
            locAuthorisation = False   
        # 2. If the balance is lower than the minimum allowed 
        if self.balance < self.minimumBalance : 
            locAuthorisation = False 
        # 3. If the profit made is higher than the maximum profit 
        if self.getProfit(option = "balance") > self.maximumProfit : 
            locAuthorisation = False 
        # 4. If the drawDown is higher than the maximum drawdown 
        if self.currentDrawDown >= self.maximumDrawDown : 
            locAuthorisation = False 
        # 5. If the consecutive losses amount is higher than the maximum consecutive losses 
        if self.currentValueLossSerie >= self.maximumConsecutiveLoss : 
            locAuthorisation = False 
        # 6. If the consecutive gains amount is higher than the maximum consecutive gains 
        if self.currentValueGainSerie >= self.maximumConsecutiveGain : 
            locAuthorisation = False 
        # 7. If the consecutive gains number is higher than the maximum number of consecutive gains 
        if self.currentMaximumNumberOfConsecutiveGains >= self.maximumNumberOfConsecutiveGains : 
            locAuthorisation = False 


        # We associate the trading authorisation to the result of the security checks 
        self.tradeAuthorisation = locAuthorisation

    def checkMarginMinimum(self) : 
        # If the margin level is below the minimum rate 
        # we close all the worse positions until the margin level becomes 
        # higher than this minimum 
        while self.marginLevel <= self.marginMinimum and len(self.openPositions) > 0 : 
            
            minProfit   = np.inf
            indexLowest = None 
            # We retrieve the position that has the worse profit 
            for i in range(len(self.openPositions)) : 
                if self.openPositions[i].profit < minProfit : 
                    minProfit   = self.openPositions[i].profit 
                    indexLowest = i
            
            # We close this position 
            if indexLowest is not None : 
                #print ("Closed position because bad margin level")
                self.closePosition(self.openPositions[indexLowest].symbol, self.openPositions[indexLowest])
            
            

    ###################################################################
    # Evolving parameters functions 
    ###################################################################
    def getProfit(self, option = "balance") : 
        """ 
        Returns the portfolio profit. 
        If option = balance : returns balance (t) - balance (t = 0) 
        If option = margin  : return availableMargin (t) - balance (t) 
        """
        if option == "balance" : 
            return self.balance - self.initialDeposit 
        if option == "margin" : 
            return self.availableMargin - self.balance 

    def getMaxDrawDown(self) :
        """ 
        Returns the current drawdown value in percentage. 
        """
        diff = (max(self.equityCurve) - self.equityCurve[-1])/max(self.equityCurve)
        if diff > self.currentDrawDown : 
            return diff 
        else : 
            return self.currentDrawDown

    ###################################################################
    # Parametric functions. Cannot be used at an higher level 
    ###################################################################
    def createOrder(self, 
                    symbolName = None,  
                    action     = "long",      # "long" or "short"
                    orderType  = "MKT",       # Order kind : "MKT", "MIT", "LMT"
                    volume     = 0.1,         # Volume 
                    stoploss   = None,          
                    takeprofit = None, 
                    lmtPrice   = None, 
                    auxPrice   = None ) : 


        # We create the ORDER parent object 
        orderParent           = ORDER() 
        orderParent.symbolName=symbolName
        orderParent.action    = action 
        orderParent.orderType = orderType 
        orderParent.volume    = volume 
        orderParent.lmtPrice  = lmtPrice

        # We create the ORDER stoploss object 
        orderSL           = ORDER()
        orderSL.symbolName= symbolName
        orderSL.parentID  = orderParent.orderID 
        orderSL.action    = "short" if orderParent.action == "long" else "long" 
        orderSL.orderType = "LMT" 
        orderSL.volume    = volume 
        orderSL.lmtPrice  = stoploss
        
        # We create the ORDER takeprofit object 
        orderTP           = ORDER()
        orderTP.symbolName= symbolName
        orderTP.parentID  = orderParent.orderID 
        orderTP.action    = "short" if orderParent.action == "long" else "long" 
        orderTP.orderType = "MIT" 
        orderTP.volume    = volume 
        orderTP.lmtPrice  = takeprofit

        # We place these orders in the pending order list 
        self.pendingOrders.append(orderParent)
        self.pendingOrders.append(orderSL)
        self.pendingOrders.append(orderTP)

        return [orderParent, orderSL, orderTP]

    
    def executeOrder(self, 
                     symbol,           # SYMBOL object
                     order,            # ORDER object to execute  
                     type   = "open"   # Two kind of orders execution : "open" (a position), "close" (a position)
                     ) :      

        if type == "open" and self.checkExecuteOrder(symbol, order) : 

            # We first calculate the request margin to operate 
            type = symbol.marginRequestMethod 
            contractSize = symbol.contractSize
            openPrice    = symbol.askprice if order.action == "long" else symbol.bidprice
            tickPrice    = None 
            tickSize     = None 
            leverage     = self.leverage

            requestMargin = financialTools.requestMargin(type, 
                                                        order.volume, 
                                                        contractSize, 
                                                        openPrice, 
                                                        tickPrice, 
                                                        tickSize, 
                                                        leverage) 
            
            #print ("request margin = ",requestMargin)
            

            # We secondly calculate the total price
            totalPrice = requestMargin*self.leverage 

            # Third we calculate the loan made by the broker 
            brokerLoan = totalPrice - requestMargin 

            # Fourth, we calculate the transaction fees 
            #commission = financialTools.transactionFees(symbol, 
            #                                            "BUY", 
            #                                            volume)
            commission = 0. 

            if (requestMargin + commission > self.availableMargin) : 
                return
        
            # Fifth, we operate the transaction 
            self.balance         -= requestMargin + commission
            self.usedMargin      += requestMargin 
            self.availableMargin -= requestMargin + commission
            self.marginLevel      = np.divide(self.availableMargin, self.usedMargin)*100.

            # Sixth, we edit the ORDER object 

            # Attributes filled at execution 
            order.totalPrice    = totalPrice  
            order.requestMargin = requestMargin 
            order.brokerLoan    = brokerLoan 
            order.commission    = commission 
            order.executionPrice= openPrice 
            order.executionDate = symbol.time
            order.comment       = ""

            # Execution status
            order.executed      = True  

            # Seventh, we create a POSITION object and move the placed order in the executed list 
            pendingOrderIndex = None 
            for i in range(len(self.pendingOrders)) : 
                if order.orderID == self.pendingOrders[i].orderID :  
                    pendingOrderIndex = i 
            if pendingOrderIndex is not None : 
                self.executedOrders.append(self.pendingOrders[pendingOrderIndex])
                del self.pendingOrders[pendingOrderIndex]
            

            locPos                = POSITION() 
            locPos.orderID        = order.orderID
            locPos.symbol         = symbol.symbolName 
            locPos.volume         = order.volume 
            locPos.action         = order.action 
            locPos.executionPrice = openPrice 
            locPos.executionDate  = symbol.time 
            locPos.brokerLoan     = brokerLoan 
            locPos.requestMargin  = requestMargin

            locPos.possibleClosePrice = symbol.bidprice if order.action == "long" else symbol.askprice
            locPos.possibleCloseDate  = symbol.time 
            absoluteProfit            = locPos.possibleClosePrice*order.volume*symbol.contractSize - brokerLoan - requestMargin
            absoluteProfit_           = absoluteProfit if order.action == "long" else -absoluteProfit
            locPos.profit             = absoluteProfit_

            locPos.comment            = "" 

            self.openPositions.append(locPos)

        if type == "close" : 

            if order.parentID is not None : 

                # We retrieve the position associated to the order 
                openPositionIndex = None 
                for i in range(len(self.openPositions)) : 
                    if self.openPositions[i].orderID == order.parentID : 
                        openPositionIndex = i 
                
                if openPositionIndex is None : 
                    print ("Error when retrieving the position associated to the closing order")
                else : 

                    order.executed = True

                    self.openPositions[openPositionIndex].possibleClosePrice = symbol.bidprice if order.action == "short" else symbol.askprice
                    self.openPositions[openPositionIndex].possibleCloseDate  = symbol.time 
                    absoluteProfit = self.openPositions[openPositionIndex].possibleClosePrice*order.volume*symbol.contractSize - self.openPositions[openPositionIndex].brokerLoan - self.openPositions[openPositionIndex].requestMargin
                    absoluteProfit_ = absoluteProfit if order.action == "short" else - absoluteProfit 
                    self.openPositions[openPositionIndex].profit = absoluteProfit_
                    
                    self.openPositions[openPositionIndex].closed = True 

                    # We operate the transaction 
                    requestMargin = self.openPositions[openPositionIndex].requestMargin
                    commission = 0. 

                    self.balance         += absoluteProfit_ + requestMargin + commission
                    self.usedMargin      -= requestMargin #+ absoluteProfit_ 
                    self.availableMargin += requestMargin #+ absoluteProfit_ 
                    self.marginLevel      = np.divide(self.availableMargin, self.usedMargin)*100.

                    # We move the closed order and order associated pending order 
                    self.closedPositions.append(self.openPositions[openPositionIndex]) 
                    del self.openPositions[openPositionIndex]

                    isOrder = True
                    while (isOrder) : 
                        pendingOrderIndex = None 
                        for i in range(len(self.pendingOrders)) : 
                            if order.parentID == self.pendingOrders[i].parentID : 
                                pendingOrderIndex = i 
                        if pendingOrderIndex is not None : 
                            self.executedOrders.append(self.pendingOrders[pendingOrderIndex])
                            del self.pendingOrders[pendingOrderIndex]
                        if pendingOrderIndex is None : 
                            isOrder = False 

                    # We update some statistics of the porfolio

                    # 1. We update the equity curve 
                    self.equityCurve.append(self.equity)

                    # 2. We update the drawdown calculation 
                    self.currentDrawDown = self.getMaxDrawDown() 

                    # 3. We update the current value loss serie 
                    if absoluteProfit_ < 0 : 
                        self.currentValueLossSerie += -absoluteProfit_ 
                        self.currentValueGainSerie = 0. 
                        self.currentMaximumNumberOfConsecutiveGains = 0
                    if absoluteProfit_ > 0 : 
                        self.currentValueGainSerie += absoluteProfit_ 
                        self.currentValueLossSerie = 0. 
                        self.currentMaximumNumberOfConsecutiveGains += 1

            else : 
                print ("Error. The order has no parent order")

                
 
            
    def checkPendingOrder(self, 
                          symbol, 
                          order) : 
        
        openPositionIndex = None 
        for i in range(len(self.openPositions)) : 
            if self.openPositions[i].orderID == order.orderID or self.openPositions[i].orderID == order.parentID :
                openPositionIndex = i 

        if openPositionIndex is not None : 
            self.updatePosition(symbol, self.openPositions[openPositionIndex])
            self.updatePortfolio()
        else : 
            pass 
        
        # If the order is the child of another order 
        if order.parentID is not None : 
            # Check if the parent of the order have been executed 
            isParentExecuted = False  
            for parent in self.openPositions : 
                if parent.orderID == order.parentID : 
                    isParentExecuted = True 
            if isParentExecuted : 
                # We check the current pending order 
                if order.orderType == "MIT": 
                    # Execute the order if : 
                    if order.action == "short" and symbol.askprice >= order.lmtPrice : 
                        self.executeOrder(symbol, order, type = "close") # Here we actualise the portfolio properties 


                    if order.action == "long" and symbol.bidprice <= order.lmtPrice : 
                        self.executeOrder(symbol, order, type = "close") # Here we actualise the portfolio properties 

                    
                if order.orderType == "LMT" : 
                    # Execute the order if : 
                    if order.action == "short" and symbol.askprice <= order.lmtPrice : 
                        self.executeOrder(symbol, order, type = "close") # Here we actualise the portfolio properties 

                    if order.action == "long" and symbol.bidprice >= order.lmtPrice : 
                        self.executeOrder(symbol, order, type = "close") # Here we actualise the portfolio properties 

        # If the order is a parent order 
        else : 
            if order.orderType == "MKT" : 
                self.executeOrder(symbol, order) # Here we actualise the portfolio properties 

            if order.orderType == "MIT" : 
                # Execute the order if : 
                if order.action == "short" and symbol.askprice >= order.lmtPrice : 
                    self.executeOrder(symbol, order) # Here we actualise the portfolio properties 


                if order.action == "long" and symbol.bidprice <= order.lmtPrice : 
                    self.executeOrder(symbol, order) # Here we actualise the portfolio properties 

            if order.orderType == "LMT" : 
                # Execute the order if : 
                if order.action == "short" and symbol.askprice <= order.lmtPrice : 
                    self.executeOrder(symbol, order) # Here we actualise the portfolio properties 

                if order.action == "long" and symbol.bidprice >= order.lmtPrice : 
                    self.executeOrder(symbol, order) # Here we actualise the portfolio properties 

    #########################################################################################
    # UPDATE FUNCTIONS
    #########################################################################################

    def updatePosition(self, 
                       symbol, 
                       position) : 

        position.possibleClosePrice =  symbol.bidprice if position.action == "long" else symbol.askprice 
        position.possibleCloseDate  =  symbol.time 
        absoluteProfit              =  position.possibleClosePrice*position.volume*symbol.contractSize - position.brokerLoan - position.requestMargin
        absoluteProfit_             =  absoluteProfit if position.action == "long" else -absoluteProfit
        position.profit             =  absoluteProfit_



    def updatePortfolio(self) :
        for position in self.openPositions : 
             self.availableMargin = self.balance + position.profit 
             self.marginLevel = np.divide(self.availableMargin, self.usedMargin)*100.
             self.equity      = self.availableMargin + self.usedMargin



    def update(self) : 

        # 1. We first update all the currently active positions 
        for key in list(self.symbols.keys()) : 
            symbol = self.symbols.get(key)
            for position in self.openPositions : 
                self.updatePosition(symbol, position)

        # 2. We then update the available margin and the margin level 
        self.updatePortfolio()
        
        # 3. We check for the security systems 
        self.tradingAuthorisation()
        self.checkMarginMinimum()

        # 4. We check for the pending orders 
        for key in list(self.symbols.keys()) : 
            symbol = self.symbols.get(key) 
            for order in self.pendingOrders : 
                if order.symbolName == key : 
                    self.checkPendingOrder(symbol, order) 
        
        # 5. We update the available margin and the margin level another time 
        self.updatePortfolio()
        
        # 6. We calculate the statistics 
        # No needs for instance. Statistics are calculated at each position close 

        # 7. We check again for the security systems 
        self.tradingAuthorisation()
        self.checkMarginMinimum()


    ###################################################################
    # Historical data price functions  
    ###################################################################
    def setHistoricalData(self, historicalData) : 
        """ 
        Function that set ...
        """ 

        self.historicalDataPrice = historicalData

    def getHistoricalData(self, symbolName, dateIni, dateEnd, timeframe, onlyOpen = True) : 
        """ 
        Function that get ...
        """ 
        # Case where the timeframe is provided as 0 
        if timeframe == 0 : 
            timeframe = self.historicalDataTimeframe


        if not onlyOpen : 

            historicalData = self.historicalDataPrice.get(symbolName)
            df = pd.DataFrame(historicalData)

            # Timeframe conversion 
            # Here from 00:00 -> 23:59 
            if timeframe != self.historicalDataTimeframe : 
                pass 

            if (type(dateIni) == dt.datetime(2020, 1, 10, 10, 10) and type(dateIni) == type(dateEnd)) : 
                df = df.set_index("date") 
                df = df[dateIni : dateEnd]
            if (type(dateIni) == int and type(dateIni) == type(dateEnd)) : 
                if dateEnd == 0: 
                    df = df[-dateIni:]
                else : 
                    df = df[-dateIni:-dateEnd]
            dictDf = df.to_dict(orient="list")
            for i in range(len(dictDf.get("date"))) : 
                dictDf.get("date")[i] = dictDf.get("date")[i].to_pydatetime()
            
            return dictDf 
        
        else : 

            historicalData = self.historicalDataPrice.get(symbolName)

            # !!! This step have to be optimized !!! (too slow)
            historicalData_ = dict()
            for key in list(historicalData.keys()) : 
                historicalData_.update({key : list()})
            for i in range(len(historicalData.get("market status"))) : 
                if historicalData.get("market status")[i] == "open" : 
                    for key in list(historicalData_.keys()) : 
                        historicalData_.get(key).append(historicalData.get(key)[i])

            # Timeframe conversion 
            # Here taking account for breaks ...  
            if timeframe != self.historicalDataTimeframe : 
                pass 

            df = pd.DataFrame(historicalData_)
            if (type(dateIni) == dt.datetime(2020, 1, 10, 10, 10) and type(dateIni) == type(dateEnd)) : 
                df = df.set_index("date") 
                df = df[dateIni : dateEnd]
            if (type(dateIni) == int and type(dateIni) == type(dateEnd)) : 
                if dateEnd == 0: 
                    df = df[-dateIni:]
                else : 
                    df = df[-dateIni:-dateEnd]
            dictDf = df.to_dict(orient="list")
            for i in range(len(dictDf.get("date"))) : 
                dictDf.get("date")[i] = dictDf.get("date")[i].to_pydatetime()
            
            return dictDf 


        #if not onlyOpen : 
        #    return dictDf
        #else : 
        #    dictDf_ = dict()
        #    for key in list(dictDf.keys()) : 
        #        dictDf_.update({key : list()})
        #    for i in range(len(dictDf.get("market status"))) : 
        #        if dictDf.get("market status")[i] == "open" : 
        #            for key in list(dictDf_.keys()) : 
        #                dictDf_.get(key).append(dictDf.get(key)[i])
        #    return dictDf_
                    


    def getLastPrice(self, symbolName) : 
        """ 
        Function that allows to get the last price info 
        """ 
        price = {
            "askopen" : self.symbols.get(symbolName).askopen, 
            "askhigh" : self.symbols.get(symbolName).askhigh, 
            "asklow"  : self.symbols.get(symbolName).asklow, 
            "askclose": self.symbols.get(symbolName).askclose, 
            "askprice": self.symbols.get(symbolName).askprice,
            "bidopen" : self.symbols.get(symbolName).bidopen, 
            "bidhigh" : self.symbols.get(symbolName).bidhigh, 
            "bidlow"  : self.symbols.get(symbolName).bidlow, 
            "bidclose": self.symbols.get(symbolName).bidclose,
            "bidprice": self.symbols.get(symbolName).bidprice, 
            "date"    : self.symbols.get(symbolName).time, 
            "volume"  : self.symbols.get(symbolName).volume, 
            "market state" : self.symbols.get(symbolName).marketState   
        }
        return price 