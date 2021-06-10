#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 


def requestMargin(
    type, 
    volume, 
    contractSize, 
    openPrice, 
    tickPrice, 
    tickSize, 
    leverage
) : 
    if type == "Forex" : 
        return volume*contractSize*openPrice/leverage 
    if type == "Stock" : 
        return volume*contractSize*openPrice/leverage 
    if type == "CFD"   : 
        return volume*contractSize*openPrice/leverage 
    if type == "CFD-Index" : 
        return volume*contractSize/openPrice/tickPrice/tickSize/leverage

def transactionFees(
    symbol, 
    transactionType, 
    volume
) : 
    if transactionType == "BUY" : 
        exchangeFee = symbol.exchangeLong 
    if transactionType == "SELL" : 
        exchangeFee = symbol.exchangeShort

    if symbol.exchangeType == "Point" : 
        subFee = exchangeFee*10**(-symbol.precision)
    if symbol.exchangeType == "Percentage" : 
        subFee = exchangeFee*0.01 

    contractSize = symbol.contractSize
    commission = contractSize*subFee*volume 

    return commission 




