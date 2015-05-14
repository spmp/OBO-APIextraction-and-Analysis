#!/usr/bin/env python3
"""
Tools for generating and working with pairwise comparison of models.

"""
import argparse, pathlib, re, glob
import pandas as pd
import numpy as np
import CategoryToNumberAssignment as c2n
#import OBOModelling as oboM

def generateHeadings(OffenceArray=c2n.offcat):
    """Generate the pairwise comparisons on an array, OffenceArray
    
    Returns
    -------
        array of strings
    """
    OffenceArrayLen=len(OffenceArray)
    HeadingArray=[]
    
    for ci,C in enumerate(OffenceArray):
        for P in range(ci+1,OffenceArrayLen):
            HeadingArray.append(c2n.upcaseFirstLetter(C)
                                +c2n.upcaseFirstLetter(OffenceArray[P]))
    
    return HeadingArray

def generatePairwiseTV(ModelFrame,OffenceArray=c2n.offcat,
                       PunishmentArray=['NotGuilty']+c2n.puncat):
    """Generate a DataFrame of TV on pairs of Offences
    
    Paramters
    ---------
    ModelFrame : pandas DataFrame
        DataFrame should contain model DataFrame
    OffenceArray : array of strings
        An array containing offences
    """
    OffenceArrayLen=len(OffenceArray)
    PunLen=len(PunishmentArray)
    PairWiseTV = pd.DataFrame(index=ModelFrame.index,
                              columns=generateHeadings(OffenceArray=OffenceArray))
    
    #Not the most efficient, but more readable
    # Iterate of categories, oi with remaining categories pj
    PairwiseIndex=0
    for i in range(0,OffenceArrayLen):
        for j in range(1,OffenceArrayLen-i):
            #This captures all n choose 2 posibilities
            PairWiseTV.iloc[:,PairwiseIndex] = 0.5*(
                ModelFrame.iloc[:,i*PunLen:(i+1)*PunLen+1] -
                ModelFrame.shift(-(j*PunLen),axis=1).iloc[:,i*PunLen:(i+1)*PunLen+1]
                ).abs().sum(axis=1)
            #print(PairwiseIndex)
            PairwiseIndex += 1
            #HeadingArray.append(C+OffenceArray[P])
    #def _categorySlice(n):
        #return ModelFrame.iloc[:,n*PunishmentArrayLen:
                                   #(n+1)*PunishmentArrayLen]
    
    #return _categorySlice(1)
    return PairWiseTV

##REcording ipython script stuff for future reference:
#PairwiseTVs = [ oboP.generatePairwiseTV(oboM.loadMod('Data/Models/OBOModelDependentCat'+str(delta)+'yrNoGender.csv')) for delta in [1,2,5,10,50]]
#PairwiseTVFs = [ oboP.generatePairwiseTV(oboM.loadMod('Data/Models/OBOModelDependentCat'+str(delta)+'yrFemale.csv')) for delta in [1,2,5,10,50]]
#PairwiseTVMs = [ oboP.generatePairwiseTV(oboM.loadMod('Data/Models/OBOModelDependentCat'+str(delta)+'yrMale.csv')) for delta in [1,2,5,10,50]]
##Reduce the number of rows
#PairwiseTVsH = [ PairwiseTVs[g].iloc[::delta,:] for g,delta in enumerate([1,2,5,10,50])]
#PairwiseTVsHF = [ PairwiseTVFs[g].iloc[::delta,:] for g,delta in enumerate([1,2,5,10,50])]
#PairwiseTVsHM = [ PairwiseTVMs[g].iloc[::delta,:] for g,delta in enumerate([1,2,5,10,50])]

#for g,delta in enumerate([1,2,5,10,50]):
    #PairwiseTVsH[g]['Average'] = PairwiseTVsH[g].sum(axis=1)/PairwiseTVsH[g].shape[1]
    #PairwiseTVsH[g].to_csv('Data/Pairwise/OBOCatPairwiseNoGender'+str(delta)+'yr.csv')
    #PairwiseTVsHF[g]['Average'] = PairwiseTVsHF[g].sum(axis=1)/PairwiseTVsHF[g].shape[1]
    #PairwiseTVsHF[g].to_csv('Data/Pairwise/OBOCatPairwiseFemale'+str(delta)+'yr.csv')
    #PairwiseTVsHM[g]['Average'] = PairwiseTVsHM[g].sum(axis=1)/PairwiseTVsHM[g].shape[1]
    #PairwiseTVsHM[g].to_csv('Data/Pairwise/OBOCatPairwiseMale'+str(delta)+'yr.csv')

#And some functions for getting the whole grid graph going....
def matricise(ModFrame):
    """Add indices etc. in a strange format to our data for surface plotting"""
    #Create a new working frame:
    Matricised = ModFrame.mul(1)
    #Add a dummy column:
    #Matricised['Dummy'] = '0\\\\'
    #Insert the first column
    Matricised.insert(0,'X00',0)
    #Iterate over rows addinf required data
    for index,col in enumerate(ModFrame.columns):
        xCoord = index//7
        yCoord = index%7
        #Insert the coordinates before the column:
        #Y coord
        Matricised.insert(Matricised.columns.get_loc(col),'Y'+str(xCoord)+str(yCoord),yCoord)
        Matricised[col]=Matricised[col].map(str)+'\\\\'+str(xCoord)
        #Add the dummy coords:
        if (index+1)%7 is 0:
            #Add column after the last with y coord
            Matricised.insert(Matricised.columns.get_loc(col)+1,'Y'+str(xCoord)+str(yCoord+1),yCoord+1)
            #Add dummy data two frames after
            Matricised.insert(Matricised.columns.get_loc(col)+2,'D'+str(xCoord)+str(yCoord+1),'0\\\\'+str(xCoord+1))
    #Add a full column of dummy data:
    for dummy in range(0,7):
        Matricised['DR'+str(dummy)] = dummy
        Matricised['DDR'+str(dummy)] = '0\\\\9'
    Matricised['DR7'] =  7
    Matricised['DDR7']= 0
    return Matricised
#oboP.matricise(CatMod.iloc[55:56]).to_csv('/tmp/test.csv',line_terminator='\\\\',index=False, header=False,quoting=None,quotechar=' ')

def matriciseSubCat(ModFrame):
    """Add indices etc. in a strange format to our data for surface plotting"""
    #Create a new working frame:
    Matricised = ModFrame.mul(1)
    #Add a dummy column:
    #Matricised['Dummy'] = '0\\\\'
    #Insert the first column
    Matricised.insert(0,'X00',0)
    #Iterate over rows addinf required data
    for index,col in enumerate(ModFrame.columns):
        xCoord = index//55
        yCoord = index%28
        #Insert the coordinates before the column:
        #Y coord
        Matricised.insert(Matricised.columns.get_loc(col),'Y'+str(xCoord)+str(yCoord),yCoord)
        Matricised[col]=Matricised[col].map(str)+'\\\\'+str(xCoord)
        #Add the dummy coords:
        if (index+1)%28 is 0:
            #Add column after the last with y coord
            Matricised.insert(Matricised.columns.get_loc(col)+1,'Y'+str(xCoord)+str(yCoord+1),yCoord+1)
            #Add dummy data two frames after
            Matricised.insert(Matricised.columns.get_loc(col)+2,'D'+str(xCoord)+str(yCoord+1),'0\\\\'+str(xCoord+1))
    #Add a full column of dummy data:
    for dummy in range(0,28):
        Matricised['DR'+str(dummy)] = dummy
        Matricised['DDR'+str(dummy)] = '0\\\\57'
    Matricised['DR28'] =  28
    Matricised['DDR28']= 0
    return Matricised
#oboP.matricise(CatMod.iloc[55:56]).to_csv('/tmp/test.csv',line_terminator='\\\\',index=False, header=False,quoting=None,quotechar=' ')
        