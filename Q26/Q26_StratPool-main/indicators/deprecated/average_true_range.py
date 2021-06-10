#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:54:43 2020

@author: lbrahimi
"""

import matplotlib.pyplot as plt 



class TR : 
    def __init__(self, high, low, close) : 
        """
        Initialisation du TR

        Parameters
        ----------
        high : Liste de float
            Contient la valeur haute des bougies
        low : Liste de float
            Contient la valeur basse des bougies
        close : Liste de float
            Contient le close des bougies

        Returns
        -------
        None.

        """
        tr = []
        for ii in range(1, len(high)) : 
            tr.append(max([(high[ii] - low[ii]), abs(high[ii] - close[ii-1]), abs(low[ii] - close[ii-1])]))
        self.value = tr
        
    def plot(self, xsize = 8, ysize = 6) : 
        """
        Fonction permettant de tracer l'indicateur TR

        Parameters
        ----------
        xsize : TYPE, optional
            DESCRIPTION. The default is 8.
        ysize : TYPE, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(xsize, ysize))
        plt.plot(self.value, ls='-', c="black", lw = 2)
        plt.show()
        plt.close()
        
    def update(self, high_i, low_i, close_im) : 
        """
        Fonction permettant d'updater l'indicateur TR

        Parameters
        ----------
        high_i : float
            Dernier high
        low_i : float
            Dernier low
        close_im : float
            Avant dernier close

        Returns
        -------
        None.

        """
        self.value.append(max([(high_i - low_i), abs(high_i - close_im), abs(low_i - close_im)]))
        del self.value[0]

class ATR : 
    
    def __init__(self, tr, n = 60*24) :
        """
        Initialisation de l'ATR

        Parameters
        ----------
        tr : TYPE, objet de la classe TR ci-dessus
            DESCRIPTION : True Range.
        n : TYPE, entier
            Période de l'indicateur. The default is 60*24.

        Returns
        -------
        None.

        """
        ATR = [0]
        sTR = 0.
        for ii in range(0, n) : 
            sTR += tr.value[ii]
        ATR.append(1./n*sTR)
        for ii in range(1, len(tr.value)) : 
            ATR.append(1./n*(ATR[-1]*(n - 1) + tr.value[ii]))
        
        self.tr     = tr
        self.value  = ATR
        self.period = n
    
    def plot(self, xsize = 8, ysize = 6) :
        """
        Fonction permettant de tracer l'indicateur ATR

        Parameters
        ----------
        xsize : TYPE, optional
            DESCRIPTION. The default is 8.
        ysize : TYPE, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(xsize, ysize))
        plt.plot(self.value, ls='-', c="black", lw = 2)
        plt.show()
        plt.close()
        
    
    def update(self, high_i, low_i, close_im) : 
        """
        Fonction permettant d'updater l'indicateur ATR

        Parameters
        ----------
        high_i : float
            Dernier high
        low_i : float
            Dernier low
        close_im : float
            Avant dernier close

        Returns
        -------
        None.

        """
        self.tr.update(high_i, low_i, close_im)
        self.value.append(1./self.period*(self.value[-1]*(self.period - 1) + self.tr.value[-1]))
        del self.value[0]
    

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
#         # TR = true_range(sub_data["askhigh"], sub_data["asklow"], sub_data["askclose"])
        
#         tr = TR(sub_data["askhigh"], sub_data["asklow"], sub_data["askclose"])
#         atr = ATR(tr, n = 50)
    
#     if (ii > 101) : 
#         # atr.update(sub_data["askhigh"], sub_data["asklow"], sub_data["askclose"])
#         # atr.plot()
        
#         high_i  = sub_data["askhigh"].iloc[-1]
#         low_i   = sub_data["asklow"].iloc[-1]
#         close_im= sub_data["askclose"].iloc[-2] 
#         atr.update(high_i, low_i, close_im)
#         # tr.update(high_i, low_i, close_im)
        
#         atr.plot()
        