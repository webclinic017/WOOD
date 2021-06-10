#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

class TIMER : 

    def __init__(self, name = None) : 
        self.ini = time.time()
        self.end = None 
        if name is not None : 
            self.name = name 
        else : 
            self.name = "no name"

    def stop(self) : 
        self.end = time.time() 
        print ("TIMER : ",self.name, ", Time = ", self.end-self.ini)