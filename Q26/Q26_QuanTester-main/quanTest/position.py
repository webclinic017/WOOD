#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class POSITION : 

    def __init__(self) : 
        # Main attributes 
        self.orderID            = None      # ID of the parent order that generated the position
        self.symbol             = None      # Symbol name  
        self.volume             = None      # Current owned volume of the asset 
        self.action             = None      # Position direction ("long" or "short")

        self.executionPrice     = None       # Execution price at position opening  
        self.executionDate      = None       # Execution date at position opening 
        self.brokerLoan         = None       # Broker loan at position opening (to be reimbursed) 
        self.requestMargin      = None       # Margin paid by the user to open this position
        self.possibleClosePrice = None       # Potential price if closing now 
        self.possibleCloseDate  = None       # Potential date if close now 
        self.profit             = None       # Potential profit if close now 

        self.comment            = None       # Comment 

        self.closed             = False      # Is the position closed 

        
