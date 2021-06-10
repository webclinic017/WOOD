#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:30:43 2020

@author: lbrahimi
"""

import matplotlib.pyplot as plt 


class EMA : 
    
    
    def __init__(self, y, period = 20) : 
        

        alpha = 2./(period + 1)
        # elif alphaChoice == "TradingView" : 
        #     # Modif for stefan and theo. Alpha from RMA from 
        #     # TradingView
        #     # See : https://www.tradingview.com/pine-script-reference/v4/#fun_rma
        #     alpha = 1./period 
        ema_temp = [y[0]]
        
        for ii in range(1, len(y)) : 
            ema_temp.append(alpha*y[ii] + (1-alpha)*ema_temp[-1])
        
        self.alpha  = alpha 
        self.value  = ema_temp 
    

    def plot(self, xsize = 8, ysize = 6) : 
        """
        Fonction permettant de tracer l'indicateur EMA

        Parameters
        ----------
        xsize : INT, optional
            DESCRIPTION. The default is 8.
        ysize : INT, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(xsize, ysize))
        plt.plot(self.value, ls='-', c="black", lw = 2)
        plt.show()
        plt.close()
    
    
    def update(self, last_y) : 
        ema_temp = self.value 
        alpha    = self.alpha 

        ema_temp.append(alpha*last_y + (1 - alpha)*ema_temp[-1])
        del ema_temp[0]
        
        self.value = ema_temp 
        

        

# Experimental tests 
# import datetime as dt 
# import sys 
        
# sys.path.append("../operators/")
# import dataReader as dtr 

# data = dtr.loadData(start_time = dt.datetime(2012, 3, 18, 10, 12), 
#                     stop_time  = dt.datetime(2012, 4, 25, 11, 18))


# for ii in range (101, len(data)-1) : 
#     sub_data = data[ii-100:ii]
    
#     if (ii == 101) : 
#         ema = EMA(sub_data["askclose"], period = 20)
#         ema.plot() 

#     if (ii > 101) : 
#         ema.update(sub_data["askclose"].iloc[-1])
#         ema.plot()


