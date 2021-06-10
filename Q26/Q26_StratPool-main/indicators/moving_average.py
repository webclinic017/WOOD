#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:51:29 2021

@author: loann
"""
import sys, os 
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname,"../")
sys.path.append(filename)

from indicators.dataset_formater import PRICE_AUGMENTATION 

class iMA : 
    
    def __init__(self, 
                 data,
                 timeframe     = None, 
                 ma_period     = None, 
                 ma_method     = "linear",
                 ma_shift      = 0, 
                 applied_price = "askclose", 
                 shift         = 0) : 
        
        if ((timeframe is None) or (ma_period is None)) : 
            print ("Error. A parameter haven't been defined.")
            return 
        
        # Parameters initialisation 
        self.timeframe     = timeframe
        self.ma_period     = ma_period 
        self.ma_method     = ma_method
        self.ma_shift      = ma_shift 
        self.applied_price = applied_price 
        self.shift         = shift
        
        self.value         = None 
        
        # First calculation 
        if ma_method == "linear" : 
            
            subData = PRICE_AUGMENTATION(data).data
            
            if len(subData.get("date")) < self.ma_period : 
                print ("Error. No enough data.") 
                return 
            else : 
                self.value = sum(x for x in subData.get(applied_price)[- ma_period - ma_shift : - 1 - ma_shift])/ma_period 
            
        