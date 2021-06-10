#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd 

class WRITER : 

    def sayHello(self) : 
        print ("Hello, I'm a writer")

    def writeClosedPositionsFile(self, 
                                 outputFile    = "./closedPositions.csv",
                                 onlyColumns   = [], 
                                 exceptColumns = [], 
                                 renameColumns = {"possibleClosePrice" : "closePrice", "possibleCloseDate" : "closeDate"}, 
                                 equityCurve   = True) : 

        closedPositionsList = self.portfolio.closedPositions 

        if len(closedPositionsList) > 0 : 

            keysList_ = list(closedPositionsList[0].__dict__.keys())

            keysList  = list()
            for key in keysList_ : 
                if len(onlyColumns) > 0 and len(exceptColumns) == 0: 
                    if key in onlyColumns : 
                        keysList.append(key)
                elif len(onlyColumns) == 0 and len(exceptColumns) > 0: 
                    if key not in exceptColumns : 
                        keysList.append(key)
                else : 
                    keysList.append(key)

            
            dataFile = dict() 
            for key in keysList : 
                dataFile.update({key : list()}) 
            
            if equityCurve : 
                equity = self.portfolio.equityCurve.copy()[1:len(closedPositionsList)+1]
                dataFile.update({"equity" : equity})
            
            for position in closedPositionsList : 
                for key in keysList : 
                    dataFile.get(key).append(position.__dict__.get(key))

            df = pd.DataFrame(dataFile) 

            df.rename(columns = renameColumns, inplace = True)

            df.to_csv(outputFile)


        else : 
            print ("There is no closed positions in the simulation")
