cimport cython
import numpy as np
cimport numpy as np
import pandas as pd
from partitionsets import partition
import probability_distribution as gslPDD
from ast import literal_eval

import CategoryToNumberAssignment as c2n
import OBOModelling as oboM

#Initilise partitions:
partitions = partition.Partition([c2n.upcaseFirstLetter(x) for x in c2n.offcat])
partitioN = partition.Partition(list(range(0,9)))


def partitionAIC(EmpFrame, part, OffenceEstimateFrame, ReturnDeathEstimate=False, Verbose=True):
    """Calculate AIC score between the EmpFrame and the model where offences are partitioned as `part'.
    
    Parameters:
    -----------
        EmpFrame : DataFrame
            DataFrame of emperical data, pre processed, maybe.
        part : nested list, 2 levels
            Partition formatted as:  [[0, 3], [1, 2, 6, 7], [4, 5, 8]]
        ReturnDeathEstimate : bool
            Whether to return the DeathEstimate frame
        
    Returns:
    --------
        AIC : DataFrame
            AIC scores for the given partition and EmpFrame by year (A row)
        DeathEstimatePerPart : DataFrame
            The contents look like: 
                [[0.4, 'Miscellaneous', 'RoyalOffences', 'ViolentTheft'],
                    [0.3, 'BreakingPeace', 'Kill'],
                    [0.2, 'Damage', 'Deception', 'Sexual', 'Theft']]
                Partitions with the probability of Death for each partition, sorted by most likely of Death
    """
    ##Cython cdefs
    cdef int DFlen
    cdef int puns
    cdef int DeathPun    
    
    #Define a verbose printing:
    #verboseprint = print if Verbose else lambda *a, **k: None
    
    #What sort of raw data are we dealing with
    DFlen = EmpFrame.shape[1]
    if DFlen is 18:
        #verboseprint('Death or Not') #placeholder
        puns = 2
        DeathPun = 1
    elif DFlen is 63:
        #verboseprint('Category frame') #placeholder
        puns = 7
        DeathPun = 2
    #else:
        #verboseprint('Frame type not recognised')
        
    ##Create Emperical frame count on partitions keeping original size:
    PartitionModel = pd.DataFrame(index = EmpFrame.index, columns = EmpFrame.columns)
    
    for partOf in part:
        #verboseprint('The partion is : {}'.format(partOf))
        #partOf looks like [0,2] for category 0 and 2
        #NOTE we need to transpose
        #Find number of events in this partitioning
        Grouping = [puns*x + y for x in partOf for y in list(range(0,puns))]
        PartGroupSum = EmpFrame.iloc[:,Grouping].sum(axis=1)
        #verboseprint('Grouping events in the following columns: {}, and the group sum is: {}'.format(Grouping, PartGroupSum))
        
        ##Loop through all punishments finding the conditional probability estimates
        for punish in range(0,puns):
            PunishGroup = [x*puns+punish for x in partOf]
            #verboseprint('PunishGroup for punishment {} is :{}'.format(punish,PunishGroup))
            punishPunishment = ((EmpFrame.iloc[:,PunishGroup].sum(axis = 1) + 1/puns).div(PartGroupSum+1)).values[np.newaxis].T
            PartitionModel.iloc[:,PunishGroup] = punishPunishment
            #verboseprint('Partition model: {}'.format(PartitionModel))
    
    if ReturnDeathEstimate:
        DeathProbFrame = pd.DataFrame([[part]], index=EmpFrame.index, columns=['Partition'])
        #Find the sum of the Death punishments per partition block
        for pId,block in enumerate(part):
            DeathProbFrame[pId] = PartitionModel.iloc[:,DeathPun::puns].iloc[:,block].iloc[:,0]
        #This one liner does a lot:
        # .apply(lambda row: ... ,axis=1) runs the following per row, where row is a pandas Series
        # we have a nested list comprehension with the outer list being:
        # [ ... for idx,x in enumerate(row['Partition'])] which enumerates on the 'Partition' stored in the rows
        #  [format("%6.5f"%row[idx])] takes the value from the 'column' (not a column in pd Series) idx, which is the probability of Death for the partition and formats it to fixed point of 5 places
        #  [c2n.offcatUp[X] for X in  x ] turns the offence numbers from the partition into their respective names.
        # (sorted( ...,key=lambda x: x[0], reverse=True) sorts the list of lists by the first element of the internal list
        DeathProbFrame = DeathProbFrame.apply(lambda row: sorted( [ [format("%3.2f"%(row[idx]*100))]+[c2n.offcatUp[X] for X in  x ] 
            for idx,x in enumerate(row['Partition'])],
            key=lambda x: float(x[0]), reverse=True),axis=1)
            
    #Now we must multiply by the Offence Estimate for P(o_i,p_j) = P(o_i)*P(p_j,^o_i)
    PartitionModel = PartitionModel*OffenceEstimateFrame
    
    #verboseprint('The Partition Model is :\n{}'.format(PartitionModel))
    #Now we find the Log-likelihood
    ll = (EmpFrame*np.log(PartitionModel.convert_objects(convert_numeric=True))).sum(axis=1)
    #k = (9-1)*(len(part))*(puns-1)
    k = 8+(len(part)*6)
    AIC = 2*k-2*ll
    #verboseprint('The Log-likelihood, the k value and the AIC scores:\n {}\n{}\n{}'.format(ll,k,AIC))
    
    if ReturnDeathEstimate:
        return AIC, DeathProbFrame
    else:
        return AIC


def generateAICtable(EmpFrame, ReturnDeathEstimate=False, BlockPunishment='Death', Verbose=False):
    
    ##Cython cdefs
    cdef int DFlen
    cdef int puns
    cdef int PartitionLen
    cdef int DeathPun
    #Define a verbose printing:
    #verboseprint = print if Verbose else lambda *a, **k: None
    PartitionLen = len(partitioN)
    
    #Initialise the AICFrame
    AICFrame = pd.DataFrame(index=EmpFrame.index, columns=list(range(PartitionLen)))
    if ReturnDeathEstimate:
        DeathProbFrame = pd.DataFrame(index=EmpFrame.index)
    
    #Find the offence probabilities:
    #What sort of raw data are we dealing with
    DFlen = EmpFrame.shape[1]
    if DFlen is 18:
        print('Death or Not') #placeholder
        puns = 2
        DeathPun = 1
    elif DFlen is 63:
        print('Category frame') #placeholder
        puns = 7
        DeathPun = c2n.puncatFullUp.index(BlockPunishment)
    else:
        print('Frame type not recognised')
        return 0
        
    #Find the prob.est's for offences
    #Generate Offence grouping:
    OffenceGrouping = [ o for o in c2n.offcat for item in list(range(0,puns))]
    #print(OffenceGrouping)
    #Group the emperical frequencies into offences
    GroupedEmp = EmpFrame.groupby(OffenceGrouping, axis=1, sort=False).sum()
    #Find the prob.est. with Laplace smoothing, without passing prior. a=1
    OffenceEstimates = (GroupedEmp + 1/GroupedEmp.shape[1]).div( EmpFrame.sum(axis=1) + 1, axis = 0)
    #Create an Offence table the same size as the original with the Offence estimates copied over
    OffenceEstimateFrame = pd.DataFrame(index = EmpFrame.index, columns = EmpFrame.columns)
    for offence in range(0,9):
        ColumnsRange= list(range(offence*puns,(offence+1)*puns))
        OffenceEstimateFrame.iloc[:,ColumnsRange] = OffenceEstimates.iloc[:,offence].values[np.newaxis].T
   
    
    
    #Loop through all partitions, calculating AIC score:
    if ReturnDeathEstimate:
        for p,part in enumerate(partitioN):
            print('Generating AIC score for row {} of {}'.format(p,PartitionLen))
            AICFrame[p], DeathProbFrame[p] = partitionAIC(EmpFrame, part, OffenceEstimateFrame, ReturnDeathEstimate=True,  Verbose=False)
    else:
        for p,part in enumerate(partitioN):
            print('Generating AIC score for row {} of {}'.format(p,PartitionLen))
            AICFrame[p] = partitionAIC(EmpFrame, part, OffenceEstimateFrame, Verbose=False)
            
    if ReturnDeathEstimate:
        return AICFrame, DeathProbFrame
    else:
        return AICFrame

def sortAICtable(AICFrame, DeathEstimateFrame = [], SortDeathEstimate=False, HowMany = 0, Verbose=False):
    """Sort the AIC frame results from best to worst, returning a frame of indexes.
    
    The best AIC score is found by minidx(axis=1) in the AIC frame, but we want more
    than just one result. This function returns a frame of the results of minidx, being
    indexes to the original frame which correspond to the partition numbers.
    
    Parameters:
        AICFrame : pandas DataFrame, The DataFrame containing AIC scores from `generateAICtable'
        HowMany : integer, How many results to return, 0 for all, and the number otherwise
        Verbose : bool, Whether to print debug information.
    """
    #Define a verbose printing:  
    
    AICFrameSorted = AICFrame.mul(1)
    
    #AICMax, the value to replace our minimum values with in order to remove them from the min list
    AICMax = AICFrameSorted.max().max()+10
    print('Frame maximum is less than {}'.format(AICMax))
    DFlon, DFlen = AICFrameSorted.shape
    print('Frame has {} rows and {} columns'.format(DFlon,DFlen))
    #Ensure AICFrameSorted has integer column names:
    AICFrameSorted.columns = list(range(DFlen))
    
    #Ensure NaN is not searched:
    AICFrameSorted.fillna(AICMax, inplace=True)
    
    if HowMany is 0:
        numToReturn = DFlen
        print('Sorting all results')
    else:
        numToReturn = HowMany
        print('Returning the top {} results'.format(numToReturn))
    #Create DataFrame to contain the results
    SortedIndex = pd.DataFrame( index=AICFrameSorted.index)
    
    #Loop through the AICFrameSorted finding the column index of requested number of minima
    print('Sorting frame by index')
    for SortNumber in range(0,numToReturn):
        print('Finding minima number {} of {}'.format(SortNumber,numToReturn))
        SortedIndex[SortNumber] = AICFrameSorted.idxmin(axis=1)
        #No doubt there is a _much_ more efficient way of doing this:
        #Set all the values from the current min search to AICMax
        #AICFrameSorted = AICFrameSorted.reset_index().apply(
        for row in range(0,DFlon):
            AICFrameSorted.iloc[row,int(SortedIndex[SortNumber].iloc[row])] = AICMax
            
    #Sort the vaues of the frame:
    print('Sorting Frame by value')
    SortedValues = AICFrame.mul(1).apply(lambda x:  np.sort(x),axis=1)
    
    if SortDeathEstimate:
        print('Sorting Death Estimate frame (mapping)')
        #Initialise 0 frame same size as original
        DFlon, DFlen = DeathEstimateFrame.shape
        DeathFrameSorted = pd.DataFrame(0, index=DeathEstimateFrame.index, columns=DeathEstimateFrame.columns)
        #Cell by cell fill each blah
        for row in range(DFlon):
            print('Sorting row {}'.format(row))
            DeathFrameSorted.iloc[row] = [DeathEstimateFrame.iloc[row,x] for x in SortedIndex.iloc[row]]
        return SortedIndex, SortedValues, DeathFrameSorted
    else:
        return SortedIndex, SortedValues

def mcmcOnPartitions(PartitionAIC, int N = 10000):
    """Traverse the partition network N times, returning a DataFrame with the traverses by year.
    
    Parameters
    ----------
    PartitionAIC : pandas DataFrame, DataFrame of AIC scores per partition and year as produced by generateAICtable.
    N : int, The number of times to run the MCMC traversal.
    
    Returns
    -------
    MCMCTraverse : pandas DataFrame, a DataFrame containing the partition numbers in a row of length N for every year.
    """
    cdef int DFlon, row 
    
    DFlon = PartitionAIC.shape[0]
    
    #We reduce the values of the AIC table to be a relative score to reduce underrun error
    print('Exponentiating AIC table')
    AICreduced = PartitionAIC.sub(PartitionAIC.min(axis=1),axis=0)
    AICexponentiated = np.exp(-(AICreduced/2))
    AICnormal = AICexponentiated.div(AICexponentiated.sum(axis=1), axis=0)
    
    #Generate table to hold samples
    MCMCTraverse = pd.DataFrame(columns=list(range(N)), index=PartitionAIC.index)
    
    print('Looping through rows rolling the many sided dice')
    for row in range(DFlon):
        print('Throwing the dice for row {} of {}'.format(row,DFlon))
        RandomGenerator = gslPDD.GeneralDiscreteDistribution(AICnormal.iloc[row])
        MCMCTraverse.iloc[row] =  [ RandomGenerator.get_random_element() for x in list(range(N))]
    
    return MCMCTraverse
 
def mapSortedIndexesToOriginal(FrameOfIndexes, MappingFrame, Verbose=False):
    """ From a FrameOfIndexes and a MappingFrame return a new frame with the Index values mapped back to those as produced by partions.Partion.
    
    Parameters
    ----------
        FrameOfIndexes : DataFrame
            A frame whose values are partition indexes, but the indexes are dis ordered from sorting where 0 is the best AIC scoring partition.
        MappingFrame : DataFrame
            The frame of indexes resulting from sorting the partitionAIC frame, used to map the values of Frame of Indexes back
    
    Returns
    -------
        MappedIndexFrame: DataFrame
            Frame with the indexes mapped back to those as produced by parttions.Partition
    """
    
    return  FrameOfIndexes.apply(lambda row: [ MappingFrame.loc[row.name][x] for x in row ], axis=1)
 
def findPartitionNumOfBlocks(FrameOfPartitionIndexes, MappedDeathFrame=[], mean=True):
    """Find thenumber of blocks or the mean number of blocks in a DF containing partition indexes
    
    Parameters
    ----------
    FrameOfIndexes : DataFrame
        A DF containing integers corresponding to partitions from partitioN
    mean : bool
        Whether to return the mean number of blocks (True) or not
        
    Returns
    -------
    DataFrame specified by mean(True|False)
    """
    if mean:
        ReturnFrame = pd.DataFrame( index=FrameOfPartitionIndexes.index)
        ReturnFrame['Mean blocks'] = FrameOfPartitionIndexes.apply(lambda row: np.mean([ len(partitioN[x]) for x in row ]), axis=1)
        if type(MappedDeathFrame) is not list:
            ReturnFrame['HighestDeathBlocksMean'] = MappedDeathFrame.apply(lambda row: np.mean([ len(literal_eval(x)[0][1:]) for x in row]) ,axis=1)
        return ReturnFrame
    else:
        ReturnFrame = pd.DataFrame( index=FrameOfPartitionIndexes.index)
        ReturnFrame['Num blocks'] = FrameOfPartitionIndexes.apply(lambda row: [ len(partitioN[x]) for x in row ], axis=1)
        if type(MappedDeathFrame) is not list:
            ReturnFrame['HighestDeathBlocksMean'] = MappedDeathFrame.apply(lambda row: [ len(literal_eval(x)[0][1:]) for x in row] ,axis=1)
        return ReturnFrame

def mcmcRelativePartitionCounts(MCMCTraverseSorted):
    """Return a frame with the number of times the mcmc traversal encounters the top ten and remaining partitionings"""
    MCMCNumberOFPartitionsOfNumberTen = pd.DataFrame(index=MCMCTraverseSorted.index)
    for blah in range(10):                                                                                        
        MCMCNumberOFPartitionsOfNumberTen['Part. '+str(blah)] = MCMCTraverseSorted.apply(lambda x: (x==blah).sum(),axis=1)
    MCMCNumberOFPartitionsOfNumberTen['>10'] = MCMCTraverseSorted.apply(lambda x: (x>10).sum(),axis=1)
    
    return MCMCNumberOFPartitionsOfNumberTen

def mcmcGenerateAll(PartitionAIC, MappingFrame, DeathEstimateFrame, Path='Data/Partitions/', Prefix='MCMCTraverse',Verbose=False):
    """Generate all required data for a range of N's storing all.
    
    Parameters
    ----------
    PartitionAIC : DataFrame
        DataFrame containing sorted AIC scores from sortAICtable
    MappingFrame : DataFrame
        DataFrame containing Partition Indexes from the SortedIndex of sortAICtable
    DeathEstimateFrame : DataFrame
        Sorted DeathEstimateFrame from sortAICtable
    
    Returns
    -------
        Files to Path.
        
    PartitionAIC = oboM.loadMod('Data/New/Partitions/AICtableSortedValuesProperly2yr.csv')
    MappingFrame = oboM.loadMod('Data/New/Partitions/AICtableSortedIndexProperly2yr.csv')
    DeathEstimateFrame = oboM.loadMod('Data/New/Partitions/DeathProbFrameSortedProperly2yr.csv')
    """
    #cdef int Ns
    #Ns = [1000,10000,100000]
    Ns = [1000]
    for N in Ns:
        print('Generating data for N={}'.format(N))
        print('Generating MCMC traversals for N={}'.format(N))
        MCMC = mcmcOnPartitions(PartitionAIC, N = N)
        print('Mapping relative MCMC results back to original partitionings for N={}'.format(N))
        MCMCmapped = mapSortedIndexesToOriginal(MCMC, MappingFrame)
        print('Mapping relative MCMC results to sorted DeathEstimateFrame for N={}'.format(N))
        MCMCmappedDeath = mapSortedIndexesToOriginal(MCMC,DeathEstimateFrame)
        print('Finding the mean number of blocks, and offences in the highest death probability block for N={}'.format(N))
        MCMCmeanNumBlocks = findPartitionNumOfBlocks(MCMCmapped, MappedDeathFrame=MCMCmappedDeath, mean=True)
        print('Finding the top ten mean number of partitionings and greater than ten from  N={}'.format(N))
        MCMCNumberOFPartitionsOfNumberTen = mcmcRelativePartitionCounts(MCMC)
        print('Writing out CSV files for N={}'.format(N))
        MCMC.to_csv( Path + Prefix + str(N) + '2yr.csv')
        MCMCmapped.to_csv( Path + Prefix + str(N) + 'MappedOriginalPartitionings2yr.csv')
        MCMCmappedDeath.to_csv( Path + Prefix + str(N) + 'MappedOriginalDeathFrame2yr.csv')
        MCMCmeanNumBlocks.to_csv( Path + Prefix + str(N) + 'MeanBlocksAndHighestDeathLength2yr.csv')
        MCMCNumberOFPartitionsOfNumberTen.to_csv( Path + Prefix + str(N) + 'NumberOfPartitionsOfNumberZeroToTen2yr.csv')
    #return MCMC, MCMCmapped, MCMCmappedDeath, MCMCmeanNumBlocks
    
def generateBootStrapEstimateConfidenceIntervals(EmpFrame, int N=1000, int Delta=2):
    """Generate the low and high confidence interval from bootsrtap data using Laplace smoothed model on Delta years."""
    #Initialise empty Panel to hold Bootstrapped DataFrames
    BootstrapPanel = pd.Panel(items=list(range(N)),major_axis=EmpFrame.index, minor_axis=EmpFrame.columns)
            
    DFlen = EmpFrame.shape[1]
    DFlon = EmpFrame.shape[0]
    ConfidenceInterval = 95
    ConfHigh = (100-(100 - ConfidenceInterval)/2)/100
    ConfLow = (0+ (100 - ConfidenceInterval)/2)/100
    
    #Initialise empty DataFrames for High and Low confidence bounds
    ConfLowFrame = pd.DataFrame(index=EmpFrame.index, columns=EmpFrame.columns)
    ConfHighFrame = pd.DataFrame(index=EmpFrame.index, columns=EmpFrame.columns)
    
    #Loop through every row in the DataFrame
    for row in range(DFlon):
    #for row in [27]:
    #for row in [235]:
        print('Generating bootstrap data for row {}'.format(row))
        
        #Represent the events from the row as a list, with each index occuring the number of times as the event, such that our list is as long as the number of events. We need this to pull samples from.
        EmpCountList = [ idx for idx,y in enumerate(EmpFrame.iloc[row]) for X in range(int(y)) ]
        EmpCountLen = len(EmpCountList)
        
        #Sample from EmpCountList N times
        #Initilise Empty DataFrame to hold Bootstrap data for this row
        if EmpCountLen is 0:
            RandomSampleEvents = pd.DataFrame(0, columns = list(range(N)), index=EmpFrame.columns )
        else:
            RandomSampleFrame = pd.DataFrame(np.random.randint(0,EmpCountLen,(EmpCountLen,N)), columns = list(range(N)), index=list(range(EmpCountLen)) )
            #In one giant step turn this into a frame of event counts
            RandomSampleEvents = RandomSampleFrame.applymap(lambda x: EmpCountList[x] ).apply(lambda x: x.value_counts(bins=[-2,-1]+list(range(DFlen+1)),sort=False).shift(1)[2:])
        #We must ensure that the index of the RandomSampleEvents frame is the same as BootsrapPanel munot_axis
        RandomSampleEvents.index = EmpFrame.columns
        
        BootstrapPanel.iloc[:,row,:] = RandomSampleEvents
     
    #Apply period resampling to the Panel
    print('Reampling data and creating models in Panels')
    BootstrapPanel = BootstrapPanel.apply(lambda x: oboM.generateDependentModelLaplace(x,Delta=Delta), axis=[1,2])
     
    #Find upper and lower percentiles and store them in respective DF's
    
    print('Finding lower and upper percentiles')
    ConfLowFrame  = BootstrapPanel.apply(lambda x: x.quantile(ConfLow),axis=0)
    ConfHighFrame  = BootstrapPanel.apply(lambda x: x.quantile(ConfHigh),axis=0)
    
    return ConfLowFrame, ConfHighFrame