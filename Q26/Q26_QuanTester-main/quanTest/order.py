#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

class ORDER : 

    def __init__(self) : 
        # Standard order attributes 
        self.symbolName    = None         # name of the symbpol associated to the order 
        self.orderID       = None         # Reference ID of the order 
        self.parentID      = None         # Reference ID of an order parent of this order 
        self.action        = None         # "long" or "short"
        self.orderType     = None         # "MKT" (Market), "LMT" (Limit), "MIT" (Market If Touched) 
        self.volume        = None         # Volume of the asset to be traded 
        self.lmtPrice      = None         # Particular price for LMT and MIT orders
        self.cancelDate    = None         # Date at which, if the order hasn't been executed yet, will be canceled 

        # Attributes filled at execution 
        self.totalPrice    = None  
        self.requestMargin = 0 
        self.brokerLoan    = None 
        self.commission    = None 
        self.executionPrice= None 
        self.executionDate = None 
        self.comment       = None 

        # Execution status
        self.executed      = False 

        # Initialization functions 
        self.initializeOrderID()
    
    #############################################################
    # INITIALISATION FUNCTIONS 
    #############################################################
    def initializeOrderID(self) : 
        n = 20 
        numList = "" 
        for i in range(n) : 
            numList += str(random.randint(0,9))
        self.orderID = int(numList)
    
    #############################################################
    # OTHER FUNCTIONS
    #############################################################
    

         

