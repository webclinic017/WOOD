#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 

class ANALYSIS : 



    def showEquityCurve(self, 
                        y_scale   = "linear", # See : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html
                        y_lim     = None, 
                        x_lim     = None,
                        x_label   = "# of transaction", 
                        y_label   = "Equity Curve", 
                        subCurve  = None, 
                        linestyle = "-",
                        marker    = None
                        ) : 

        y = self.portfolio.equityCurve
        
        if subCurve is not None : 
            y = y[subCurve[0] : subCurve[1]]

        fig, ax = plt.subplots()
        l1, = ax.plot(y, ls = linestyle, marker = marker)

        if y_lim is not None : 
            ax.set_ylim(y_lim[0], y_lim[1])
        if x_lim is not None : 
            ax.set_xlim(x_lim[0], x_lim[1])

        ax.set_yscale(y_scale)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label) 

        plt.show()
