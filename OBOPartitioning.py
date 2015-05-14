#!/usr/bin/env python3
"""
This module is for finding optimal partitions on punishments and offences, exhaustively.

    * Pandas is used to handle data
    * partitions are found using partition.Partitions from the 'PartitionSets' library.
        This has the advantage of giving a canonical ordering to the partitions, generated on request
    * gsl rand_discrete is used to generate MCMC traverses via CythonGSL
    
Ecosystem:
    AICtable : A table with rows of years (matches input frame) and columns of partitions
    AICtableSortedValues : The AICtable with rows sorted by AIC score
    AICtableSortedParitionNumber : The associated partition numbers of the sorted values score
    
    MCMCTraverse : Table similar to the AICtable with rows of years and N columns, one for every traverse
        The MCMCTraverse can take AICtableSortedValues as an imput such that maximally occuring scores are close to 0,
         The values can be mapped back by
"""
import pandas as pd
import numpy as np
from ast import literal_eval
from partitionsets import partition
import CategoryToNumberAssignment as c2n
import OBOModelling as oboM

#The following dependency is from CythonGSL from https://github.com/twiecki/CythonGSL
# The interfaces must be installed as:
#  sudo python3 setup_interface.py install
# in the CythonGSL directory. (you will need gcc and libgsl-dev or equivalent installed)
import probability_distribution as gslPDD

Deltas = [1,2,3,4,5,10,50,100,240]
#Initilise partitions:
partitions = partition.Partition([c2n.upcaseFirstLetter(x) for x in c2n.offcat])
partitioN = partition.Partition(list(range(0,9)))

def partitionAIC(EmpFrame, part, OffenceEstimateFrame = [], ReturnDeathEstimate=False, BlockPunishment='Death', Verbose=True):
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
    
    """
    To be clear, we are not trying to discover the true number of offence
    categories, just how many unique punishment distributions live on top of the
    catefories and how the unique distributions change.
    
    Partitioning and storing the scores as before it is the calculation of the
    estimates that is different.
    
    For each Offence/Punishment pair we find the partitioned estimate as:
    P(o_i,p_j) = P(o_i)*P(p_j|^o_i), where P(o_i) is independent of the partion,
    and P(p_j|^o_i) is the conditional prob.est. of p_j on the partitioned
    offence ^o_i.
    The number of parameters for the a partitioning is given by:
     k = (9-1)*(partition block size)*(7-1) in the full case and
     k = (9-1)*(partition block size)*(7-1) in the bernoulli case.
    The (9-1) term is constant as we are not changing the number of observed
    events, just the number of 'dice' used to determine the punishment given an
    offence.
    
    we will use Laplace smoothing without passing through the prior, with 
    alpha=1
    
    Algorithm:
        Find a static table of Offence estimates per year (9x(243/Deltat))
        Find P(p_j|^o_i), for the partition
        Copy the values from the reduced table back into the full sized table
            to maintain the same size 
        Multilply the new table by the Offence estimate table
        Calculate loglikilyhood as before
        Calculate k
        Calculate AIC
        Store row of AIC values.
    Shall we:   
    """
    #TODO: Clean up the DeathFrame part, it is now outdated and incomplete
    
    #Define a verbose printing:
    verboseprint = print if Verbose else lambda *a, **k: None
    
    #What sort of raw data are we dealing with
    DFlen = EmpFrame.shape[1]
    if DFlen is 18:
        verboseprint('Death or Not') #placeholder
        puns = 2
        DeathPun = 1
    elif DFlen is 63:
        verboseprint('Category frame') #placeholder
        puns = 7
        DeathPun = c2n.puncatFullUp.index(BlockPunishment)
    else:
        verboseprint('Frame type not recognised')
    
    if type(OffenceEstimateFrame) is list:
    #Commented out this section as it only needs to be calculated once,
    # and is better done by the 'generateAICtable' callling function.
    # Simply uncomment if testing on a single partition.
        #Find the prob.est's for offences
        verboseprint('Generating Offence groupings')
        OffenceGrouping = [ o for o in c2n.offcat for item in list(range(0,puns))]
        verboseprint(OffenceGrouping)
        #Group the emperical frequencies into offences
        GroupedEmp = EmpFrame.groupby(OffenceGrouping, axis=1, sort=False).sum()
        #Find the prob.est. with Laplace smoothing, without passing prior. a=1
        OffenceEstimates = (GroupedEmp + 1/GroupedEmp.shape[1]).div( EmpFrame.sum(axis=1) + 1, axis = 0)
        #Create an Offence table the same size as the original with the Offence estimates copied over
        OffenceEstimateFrame = pd.DataFrame(index = EmpFrame.index, columns = EmpFrame.columns)
        for offence in range(0,9):
            ColumnsRange= list(range(offence*puns,(offence+1)*puns))
            OffenceEstimateFrame.iloc[:,ColumnsRange] = OffenceEstimates.iloc[:,offence].values[np.newaxis].T
        verboseprint('The Offence estimate frame is:\n{}'.format(OffenceEstimateFrame))
   
    ##verboseprint('Processing partition {}, being: {}'.format(p,part))
    ##Create Emperical frame count on partitions keeping original size:
    PartitionModel = pd.DataFrame(index = EmpFrame.index, columns = EmpFrame.columns)
    #GroupedPartitionedOffenceFrequencies = pd.DataFrame(index=EmpFrame.index, columns = EmpFrame.columns)
    #verboseprint('The Emperical Frame is: \n{}'.format(EmpFrame))
    #verboseprint('The complete partitioning is :{}'.format(part))
    
    for partOf in part:
        verboseprint('The partion is : {}'.format(partOf))
        #partOf looks like [0,2] for category 0 and 2
        #NOTE we need to transpose
        #Find number of events in this partitioning
        Grouping = [puns*x + y for x in partOf for y in list(range(0,puns))]
        PartGroupSum = EmpFrame.iloc[:,Grouping].sum(axis=1)
        verboseprint('Grouping events in the following columns: {}, and the group sum is: {}'.format(Grouping, PartGroupSum))
        
        ##Loop through all punishments finding the conditional probability estimates
        for punish in range(0,puns):
            PunishGroup = [x*puns+punish for x in partOf]
            verboseprint('PunishGroup for punishment {} is :{}'.format(punish,PunishGroup))
            punishPunishment = ((EmpFrame.iloc[:,PunishGroup].sum(axis = 1) + 1/puns).div(PartGroupSum+1)).values[np.newaxis].T
            PartitionModel.iloc[:,PunishGroup] = punishPunishment
            verboseprint('Partition model: {}'.format(PartitionModel))
    
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
    PartitionModelPre = PartitionModel.mul(1)
    PartitionModel = PartitionModel*OffenceEstimateFrame

    verboseprint('The Partition Model is :\n{}'.format(PartitionModelPre))
    #Now we find the Log-likelihood
    ll = (EmpFrame*np.log(PartitionModel.convert_objects(convert_numeric=True))).sum(axis=1)
    #k = (9-1)*(len(part))*(puns-1)
    k = 8+(len(part)*6)
    AIC = 2*k-2*ll
    verboseprint('The Log-likelihood, the k value and the AIC scores:\n {}\n{}\n{}'.format(ll,k,AIC))
    
    if ReturnDeathEstimate:
        return AIC, DeathProbFrame
    else:
        return AIC


def generateAICtable(EmpFrame, ReturnDeathEstimate=False, Verbose=False):
    
    
    #Define a verbose printing:
    verboseprint = print if Verbose else lambda *a, **k: None
    PartitionLen = len(partitioN)
    
    #Initialise the AICFrame
    AICFrame = pd.DataFrame(index=EmpFrame.index, columns=list(range(PartitionLen)))
    if ReturnDeathEstimate:
        DeathProbFrame = pd.DataFrame(index=EmpFrame.index)
    
    #Find the offence probabilities:
    #What sort of raw data are we dealing with
    DFlen = EmpFrame.shape[1]
    if DFlen is 18:
        verboseprint('Death or Not') #placeholder
        puns = 2
        DeathPun = 1
    elif DFlen is 63:
        verboseprint('Category frame') #placeholder
        puns = 7
        DeathPun = 2
    else:
        verboseprint('Frame type not recognised')
        return 0
        
    #Find the prob.est's for offences
    #Generate Offence grouping:
    OffenceGrouping = [ o for o in c2n.offcat for item in list(range(0,puns))]
    verboseprint(OffenceGrouping)
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
            verboseprint('Generating AIC score for row {} of {}'.format(p,PartitionLen))
            AICFrame[p], DeathProbFrame[p] = partitionAIC(EmpFrame, part, OffenceEstimateFrame, ReturnDeathEstimate=True, Verbose=False)
    else:
        for p,part in enumerate(partitioN):
            verboseprint('Generating AIC score for row {} of {}'.format(p,PartitionLen))
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
    verboseprint = print if Verbose else lambda *a, **k: None
    
    AICFrameSorted = AICFrame.mul(1)
    
    #AICMax, the value to replace our minimum values with in order to remove them from the min list
    AICMax = AICFrameSorted.max().max()+10
    verboseprint('Frame maximum is less than {}'.format(AICMax))
    DFlon, DFlen = AICFrameSorted.shape
    verboseprint('Frame has {} rows and {} columns'.format(DFlon,DFlen))
    #Ensure AICFrameSorted has integer column names:
    AICFrameSorted.columns = list(range(DFlen))
    
    #Ensure NaN is not searched:
    AICFrameSorted.fillna(AICMax, inplace=True)
    
    if HowMany is 0:
        numToReturn = DFlen
        verboseprint('Sorting all results')
    else:
        numToReturn = HowMany
        verboseprint('Returning the top {} results'.format(numToReturn))
    #Create DataFrame to contain the results
    SortedIndex = pd.DataFrame( index=AICFrameSorted.index)
    
    #Loop through the AICFrameSorted finding the column index of requested number of minima
    verboseprint('Sorting frame by index')
    for SortNumber in range(0,numToReturn):
        verboseprint('Finding minima number {} of {}'.format(SortNumber,numToReturn))
        SortedIndex[SortNumber] = AICFrameSorted.idxmin(axis=1)
        #No doubt there is a _much_ more efficient way of doing this:
        #Set all the values from the current min search to AICMax
        for row in range(0,DFlon):
            AICFrameSorted.iloc[row,int(SortedIndex[SortNumber].iloc[row])] = AICMax
            
    #Sort the vaues of the frame:
    verboseprint('Sorting Frame by value')
    SortedValues = AICFrameSorted.apply(lambda x:  np.sort(x),axis=1)
    
    if SortDeathEstimate:
        verboseprint('Sorting Death Estimate frame (mapping)')
        #Initialise 0 frame same size as original
        DFlon, DFlen = DeathEstimateFrame.shape
        DeathFrameSorted = pd.DataFrame(0, index=DeathEstimateFrame.index, columns=DeathEstimateFrame.columns)
        #Cell by cell fill each blah
        for row in range(DFlon):
            DeathFrameSorted.iloc[row] = [DeathEstimateFrame.iloc[row,x] for x in SortedIndex.iloc[row]]
        return SortedIndex, SortedValues, DeathFrameSorted
    else:
        return SortedIndex, SortedValues


#Death Frame and probability per row and partitioning:
#def deathProbAndPartition( EmpFrame, OffenceEstimateFrame, row, partitionNumber):
    


##The following is for the MCMC searching on Partitions

def mcmcOnPartitions(PartitionAIC, N = 10000, Verbose=False, Sort=False):
    """Traverse the partition network N times, returning a DataFrame with the traverses by year.
    
    Parameters
    ----------
    PartitionAIC : pandas DataFrame, DataFrame of AIC scores per partition and year as produced by generateAICtable. Best to use a sorted AIC scores table for plotting.
    N : int, The number of times to run the MCMC traversal.
    
    Returns
    -------
    MCMCTraverse : pandas DataFrame, a DataFrame containing the partition numbers in a row of length N for every year.
    """
    #Define a verbose printing:
    verboseprint = print if Verbose else lambda *a, **k: None
    
    if Sort:
        PartitionAIC = PartitionAIC.apply(lambda x:  np.sort(x),axis=1)
        #verboseprint('OMG')
        
    DFlon = PartitionAIC.shape[0]
    
    #We reduce the values of the AIC table to be a relative score to reduce underrun error
    verboseprint('Exponentiating AIC table')
    AICreduced = PartitionAIC.sub(PartitionAIC.min(axis=1),axis=0)
    AICexponentiated = np.exp(-(AICreduced/2))
    AICnormal = AICexponentiated.div(AICexponentiated.sum(axis=1), axis=0)
    
    #Generate table to hold samples
    MCMCTraverse = pd.DataFrame(columns=list(range(N)), index=PartitionAIC.index)
    
    verboseprint('Looping through rows rolling the many sided dice')
    for row in range(DFlon):
        verboseprint('Throwing the dice for row {} of {}'.format(row,DFlon))
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
    Ns = [1000,10000,100000]
    #Ns = [100]
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
    

#Find the minimal intersections of a list of partitions.
def IntersectionsToRuleThemAll(row):
    #Convert row to list
    def rowTolist(row):
        from ast import literal_eval
        #print(type(literal_eval(row)))
        return [ literal_eval(x) for x in row ]
    
    def areAllEqual(items):
        return all(x == items[0] for x in items)
        
    def exPandList(List):
        return [ y for x in List for y in x]

    def intersect(A,B):
        return [val for val in A if val in B]

    def intersectiorIntersection(List,ListOfLists):
        return [ [intersect(List,x) for x in y if len(intersect(List,x)) is not 0] for y in ListOfLists]

    def intersector(l, L,Store):
        """Recursive function"""
        #Store=[]
        for d1 in l:
            R = intersectiorIntersection(d1,L)
            #print(R)
            if len(R) is 1:
                #print('Reduced to One: {}'.format(R[0]))
                for bit in R[0]:
                    Store.append(bit)
            elif areAllEqual(R):
                for bit in R[0]:
                    Store.append(bit)
            else:
                #print('Next intersector on {} and {}'.format(R[0],R[1:]))
                intersector(R[0],R[1:],Store)
    
    Liiiist = rowTolist(row)
    Store=[]
    intersector(Liiiist[0],Liiiist[1:],Store)
    
    #Store = exPandList(Store)
    Store.sort()
    #print(Store)
    return sorted(Store, key=len)

#Find the probability of death of every offence
def stripDeathProbability(DeathFrameSorted, Column=0, Total=False):
    """Find the probability of death as a punishment for each offence.
    Parameters
    ----------
    DeathFrameSorted : DataFrame
        `DeathFrameSorted' as returned by sortAICtable containting lists of offences
    Column : integer
        The column number from which to find the death prob.s in
    
    Returns
    -------
    ProbOfDeathFrame : DataFrame
        DataFrame with columns of offences and rows of dates
    """
    #Create the output DF:
    ProbOfDeathFrame = pd.DataFrame(index=DeathFrameSorted.index, columns = c2n.offcatUp )
    
    #For the given row associate the Offence with a probability of death.
    def pullPunProbFromRow(row):
        for block in literal_eval(row[Column]):
            for Offence in block[1:]:
                ProbOfDeathFrame.loc[row.name,Offence] = float(block[0])
                
    #Use apply instead of a for loop over rows.
    DeathFrameSorted.apply(pullPunProbFromRow, axis=1 )
    
    if Total:
        ProbOfDeathFrame['Total'] = ProbOfDeathFrame.sum(axis=1)
        
    return ProbOfDeathFrame

##Proper Prior
#Ssk = [1, 255, 3025, 7770, 6951, 2646, 462, 36, 1]
#Ks = list(range(10))
#Ks = [x*7 for x in Ks]
#Denom = sum([ Ssk[idx]*np.exp(-Ks[idx]) for idx in list(range(0,9))])
#SomethingElse = [np.exp(-Ks[idx])/Denom for idx in list(range(9))]
#ProperPrior = [ SomethingElse[idx]*Ssk[idx] for idx in list(range(9))]


def bootstrapMCMCprobabilityOfDeath(EmpFrame, AICFrame, N=10, MCMConPartitions = [], Delta=2, Punishment='Death'):
    """Return the high and low confidence intervals for probabiity of death base on MCMC run on partitions.
    
    """
    #NOTE: Unfinished WIP
    #Generate our own dummy data
    """
EmpFrame = oboM.changePeriod( oboM.loadCatEmp( 'Data/Raw/OBOextractNoGender1yr.csv'), Delta=2).iloc[ [30, 31, 116, 117]]
AICFrame = oboM.loadMod('Data/Partitions/AICtableSortedValuesProperly2yr.csv').iloc[[30,31,116,117],0:4]
    """
    puns = 7
    offs = 9
    PunIndex = c2n.puncatFullUp.index(Punishment)
    
    DFlon,DFlen = EmpFrame.shape
    
    OffenceEstimate = oboM.generateDependentModelLaplace(oboM.combineOnOffence(EmpFrame),Delta=Delta)
    
    #Generate MCMC data if it does not exist.
    if type(MCMConPartitions) is list:
        print('Generating MCMC on {} partitions'.format(N))
        MCMConPartitions =  mcmcOnPartitions(AICFrame, N = N, Verbose=True, Sort=False)
    
    MCMClon,MCMClen = MCMConPartitions.shape
    #Convert the partition numbers to partitions:
    #print('Converting the MCMC frame of partition numbers to partitions')
    #MCMCofPartitions = MCMCofPartitions.applymap(lambda x: oboP.partitioN[x])
        
    for row in range(DFlon):
        print('Calculating probabilites for row {} of {}'.format(row,DFlon))
        print('Looping through every MCMC partitioning')
        #Find the groupings for the counts per partition as a series
        Row = MCMCofPartitions.iloc[row]
        RowGrouping = Row.apply(lambda partId: [ x for x in oboP.partitioN.unrank_rgf(partId)[1:] for y in range(puns) ])
        #Find the punishment groupings, with 1 for the punishment and 0 else, discarding 0:
        PunishmentGrouping = Row.apply(lambda partId: [ x if y is PunIndex else 0  for x in oboP.partitioN.unrank_rgf(partId)[1:] for y in list(range(puns)) ])
        #Find the percentage probability estimates for the punishment, per block:
        RowCountsPerBlock = RowGrouping.apply(lambda group: EmpRow.groupby(group).sum())
        PunishmentCountPerBlock = RowDeathOffGrouping.apply(lambda blah: EmpRow.groupby(blah).sum()[1:])
        PunishmentEstimatesPerBlock = (PunishmentCountPerBlock/ RowGroupingCounts)*100
        #Get rid of the pesky il defined infinities
        PunishmentEstimatesPerBlock[np.isinf(PunishmentEstimatesPerBlock)] = 0
        

        #for MCMC in range(MCMClen):
            #print('Finding estimates for {} for row {}, partition {}'.format(Punishment, row, MCMC))
            
            #RowDeathOffGrouping = EmpFrame.iloc[row].apply(lambda partId: [ x if y is PunIndex else 0  for x in partitioN.unrank_rgf( MCMConPartitions.iloc[row,MCMC] )[1:] for y in list(range(puns)) ])
            #print(RowDeathOffGrouping)
    
    return OffenceEstimate, MCMConPartitions
    
    
    