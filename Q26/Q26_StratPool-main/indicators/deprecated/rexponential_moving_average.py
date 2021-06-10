#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:01:23 2020

@author: loann
"""

import matplotlib.pyplot as plt 


class RMA : 
    
    
    def __init__(self, y, period = 20) : 
        """ 
        Moving average used in RSI from TradingView. It is the exponentially weighted moving average with alpha = 1 / length
        The one from TradingView. See : https://www.tradingview.com/pine-script-reference/v4/#fun_rma
        """ 
        alpha = 1./period 
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