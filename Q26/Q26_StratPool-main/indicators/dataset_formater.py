#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:45:22 2021

@author: loann
"""

class PRICE_AUGMENTATION : 
    
    def __init__(self, data) : 
        
        self.askmedian   = list() 
        self.asktypical  = list() 
        self.askweighted = list()
        self.bidmedian   = list() 
        self.bidtypical  = list() 
        self.bidweighted = list()
        
        for i in range(len(data.get("date"))) : 
            self.askmedian.append(0.5*(data.get("askhigh")[i] + data.get("asklow")[i]))
            self.asktypical.append((data.get("askhigh")[i] + data.get("asklow")[i] + data.get("askclose")[i])/3.)
            self.askweighted.append(0.25*(data.get("askhigh")[i] + data.get("asklow")[i] + 2.*data.get("askclose")[i]))
            
            self.bidmedian.append(0.5*(data.get("bidhigh")[i] + data.get("bidlow")[i]))
            self.bidtypical.append((data.get("bidhigh")[i] + data.get("bidlow")[i] + data.get("bidclose")[i])/3.)
            self.bidweighted.append(0.25*(data.get("bidhigh")[i] + data.get("bidlow")[i] + 2.*data.get("bidclose")[i]))
        
        data.update({"askmedian"   : self.askmedian, 
                     "asktypical"  : self.asktypical, 
                     "askweighted" : self.askweighted, 
                     "bidmedian"   : self.bidmedian, 
                     "bidtypical"  : self.bidtypical, 
                     "bidweighted" : self.bidweighted})
        
        self.data = data
        
        return  
