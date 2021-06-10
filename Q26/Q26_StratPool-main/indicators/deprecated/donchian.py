#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 08:04:11 2020

@author: lbrahimi
"""
import numpy as np 
import matplotlib.pyplot as plt 


class Donchian : 
    
    def __init__(self, data, p = 20) : 
        """
        Initialisation du Donchian

        Parameters
        ----------
        data : PANDAS DATAFRAME
            Données historiques.
        p : INT, optional
            Période de l'indicateur en unité de bougie. The default is 20.

        Returns
        -------
        None.

        """
        donchian_sup = np.empty(len(data.index))
        for ii in range(len(donchian_sup)) : 
            if (ii == 0) : 
                donchian_sup[ii] = data["askhigh"].iloc[0]
            if (ii < p and ii > 0) : 
                donchian_sup[ii] = max(data["askhigh"].iloc[0:ii])
            if (ii >= p) : 
                donchian_sup[ii] = max(data["askhigh"].iloc[ii-p:ii])
    #    donchian_sup[-1] = donchian_sup[-2]
                
        donchian_inf = np.empty(len(data.index))
        for ii in range(len(donchian_inf)) : 
            if (ii == 0) : 
                donchian_inf[ii] = data["asklow"].iloc[0]
            if (ii < p and ii > 0) : 
                donchian_inf[ii] = min(data["asklow"].iloc[0:ii])
            if (ii >= p) : 
                donchian_inf[ii] = min(data["asklow"].iloc[ii-p:ii])
    #    donchian_inf[-1] = donchian_inf[-2]
        
        donchian_mean = np.empty(len(data.index))
        for ii in range(len(donchian_mean)) : 
            donchian_mean[ii] = 0.5*(donchian_sup[ii] + donchian_inf[ii])
        
        self.time   = list(data.index)
        self.inf    = list(donchian_inf)
        self.mean   = list(donchian_mean)
        self.sup    = list(donchian_sup)
        self.period = p
        
        
    def plot(self, xsize = 8, ysize = 6) : 
        plt.figure(figsize=(xsize, ysize))
        plt.plot(self.time, self.inf, lw = 1, c="black")
        plt.plot(self.time, self.mean, lw = 2, c="black")
        plt.plot(self.time, self.sup, lw = 1, c="black")
        plt.show()
        plt.close()
        
        
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
#         dc = Donchian(sub_data, p=20)
        
#     if (ii  > 101) :
#         # dc.update(sub_data["askhigh"].iloc[-1], sub_data["asklow"].iloc[-1], sub_data.index[-1])
#         dc = Donchian(sub_data, p=20)
#         dc.plot()
        
        
        
        
        
        
        
        
        
        
        
        