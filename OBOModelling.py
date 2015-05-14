#!/usr/bin/env python3
"""
Tools and code creating, comparing and analysing models (probability estimates) from the Old Bailey Online raw data, Version Two.

The module consists of three basic types of function:

 * Data loading/saving
 * Model generation
 * Model comparison and analytics

"""
#%load_ext autoreload
import argparse, pathlib, re, glob, math
import pandas as pd
import numpy as np
import CategoryToNumberAssignment as c2n


#For loading frames from file

def loadCatEmp(DataFramePath):
    """Generate a category dataframe from raw OBO extraction.
    
    Parameters:
    -----------
    DataFramePath : string
        Path to a raw OBOapiExtraction generated file
        
    Returns:
    --------
    CatEmp : pandas DataFrame
        Only the 'categories' data with date corrected and set as index
    """
    CatEmp = pd.DataFrame.from_csv(DataFramePath).loc[:,'Date':'ViolentTheftTransport']
    CatEmp['Date']=(CatEmp['Date']//10000).astype(int)
    CatEmp = CatEmp.set_index('Date')
    return CatEmp


def loadSubCatEmp(DataFramePath):
    """Generate a subcategory dataframe from raw OBO extraction.
    
    Parameters:
    -----------
    DataFramePath : string
        Path to a raw OBOapiExtraction generated file
        
    Returns:
    --------
    SubCatEmp : pandas DataFrame
        Only the 'subcategories' data with date corrected and set as index
    """
    SubCatEmp = pd.DataFrame.from_csv(DataFramePath)
    SubCatEmp['Date']=(SubCatEmp['Date']//10000).astype(int)
    SubCatEmp = SubCatEmp.set_index('Date')
    SubCatEmp = SubCatEmp.loc[:,'AnimalTheftNotGuilty':'WoundingWhipping']
    return SubCatEmp


def loadMod(DataFramePath):
    """Generate a model dataframe from a csv containing Only a category or subcategory model.

    Parameters
    ----------
    DataFramePath : string
        Path to a DataFrame csv containing only A category or subcategory model.
        This will not separate a combined csv.

    Returns
    -------
    Pandas DataFrame
    """
    try:
        Mod =  pd.DataFrame.from_csv(DataFramePath, index_col='Date')
    except:
        Mod =  pd.DataFrame.from_csv(DataFramePath)
    return Mod

# Useful tools for modifying empirical or raw data

def combineOnOffence(DataFrame):
    """Combine a CatFrame (Not subCat frame) on offences
    
    Parameters
    ----------
    DataFrame : pandas DataFrame, a Categories frame with 63 columns
    
    Returns
    -------
    OffenceFrame : pandas DataFrame, with 9 Offence Category Columns
    """
    offenceGroup = [ offence for offence in c2n.offcatUp for punishment in c2n.puncatFullUp]
    OffenceFrame = DataFrame.groupby(offenceGroup, axis=1, sort=False).sum()
    return OffenceFrame


def combineOnPunishment(DataFrame):
    """Combine a CatFrame (Not subCat frame) on offences
    
    Parameters
    ----------
    DataFrame : pandas DataFrame, a Categories frame with 63 columns
    
    Returns
    -------
    OffenceFrame : pandas DataFrame, with 9 Offence Category Columns
    """
    punishmentGroup = [ punishment for offence in c2n.offcatUp for punishment in c2n.puncatFullUp]
    PunishmentFrame = DataFrame.groupby(punishmentGroup, axis=1, sort=False).sum()
    return PunishmentFrame

def hyphenateCategories(DataFrame):
    """Hyphenate runtogether offence-punishment column labels.
    
    Parameters
    ----------
    DataFrame : pandas DataFrame, a Categories frame with 63 columns
    
    Returns
    -------
    HyphenateFrame : pandas DataFrame, with 9 Offence Category Columns
    """
    DataFrame.columns = c2n.generateCategoriesHyphenated()
    return DataFrame
    
#For generating probability estimate 'models':

def changePeriod(EmpFrame,Delta=2, MovingAverage=False):
    """Increase the sampling period of the emperical data from one to Delta years.
    
    Parameters
    ----------
    EmpFrame : pandas DataFrame
        Assume that EmpFrame is of one year emperical DataFrame from 1764-1913
    Delta : integer
        The period with which to resample over
    
    Returns:
    DataFrame
    """
    StartDate=1674
    #Check the number of rows in the EmpFrame
    EmpRows = EmpFrame.shape[0]
    
    if MovingAverage is False:
        #Create an empty list  the same length as the number of rows
        # with value arbitary larger than the number of rows
        indexGroup = [EmpRows+1]*EmpRows
        #Create a list with EmpRows/Delta(ish) unique entries the same size as EmpRows
        for row in range(0, math.ceil(EmpRows/Delta)):
            indexGroup[row*Delta:(row+1)*Delta] = [(row)*Delta+StartDate]*Delta
        #Remove extra entries
        del indexGroup[EmpRows:]
        
        #Resample the EmpFrame:
        return EmpFrame.groupby(indexGroup,sort=False).sum()
    if MovingAverage is True:
        #Use pandas internal rolling_mean. min_periods=0 gits rid of NaN at the boundaries
        return pd.rolling_mean(EmpFrame,Delta,min_periods=0,center=True)

def deathornot(CatEmp):
    """Create the raw data for death or not as punishment per offence.
    
    Use generateDependentModelLaplace(deathornot) to generate the probability estimates
    """
    
    #Grouping arrays, statically as it is easier to understand. 63 columns, combine by category for Death and Not:
    Groupings = [ c2n.upcaseFirstLetter(x)+y for x in c2n.offcat for y in ['Not']*2 + ['Death'] + ['Not']*4]
    DeathOrNotEmp = CatEmp.groupby(Groupings,axis=1,sort=False).sum()
    
    return DeathOrNotEmp

# Generate models
def generateDependentModelMLE(EmpFrame, Delta=1, MovingAverage = False):
    """Count the number of events in categories of offence and punishment for the indipendent model.
    
    Parameters:
    -----------
    EmpFrame : pandas EmpFrame
        Dataframe containing one year emperical data
    Delta : integer
        How many years in each block
    """
    #resample EmpFrame
    if Delta > 1:
        EmpFrame = changePeriod(EmpFrame,Delta=Delta, MovingAverage = MovingAverage)
    
    return EmpFrame.div(EmpFrame.sum(axis=1),axis=0).fillna(0)

def generateDependentModelLaplace(EmpFrame, Delta=1, MovingAverage = False):
    """Generate the probability estimate ('model') using Laplace smoothing with update.
    
    Parameters:
    -----------
    EmpFrame : pandas EmpFrame
        Dataframe containing one year emperical data
    Delta : integer
        How many years in each block
    """
    #Resample the dataframe by Delta:
    if Delta > 1:
        EmpFrame = changePeriod(EmpFrame,Delta=Delta, MovingAverage = MovingAverage)
        #With moving average, we need to set Delta = 1 as every row exists now,
        # so row-Delta needs to be the previous year by name.
        if MovingAverage is True: Delta = 1
    
    #Generate empty dataframe of the same size as the one give:
    ModelFrame = pd.DataFrame(columns=EmpFrame.columns, index=EmpFrame.index)
    DFlen = EmpFrame.shape[1]
    DFnumRows = EmpFrame.shape[0]
    #For the first row set alpha = 1
    #ModelFrame.iloc[0] = (EmpFrame.iloc[0]+1).div(EmpFrame.iloc[0].sum() + DFlen)
    ModelFrame.iloc[0] = (EmpFrame.iloc[0]+1/DFlen).div(EmpFrame.iloc[0].sum() + 1)
    #Loop through Rows using the previous row as alpha
    for row in range(1,DFnumRows):
        ModelFrame.iloc[row] = (EmpFrame.iloc[row] + ModelFrame.iloc[row-1]).div( EmpFrame.iloc[row].sum() + 1 )
    
    #Return DataFrame ensuring correct data type (without this we get 'float' instead of 'numpy...'
    return ModelFrame.convert_objects(convert_numeric=True)

def generateIndependentModel(EmpFrame, Delta=1, type='Laplace', MovingAverage=False):
    """Generate an independent probability estimate ('model') using Laplace smoothing or MLE.
    %TODO: Do this with proper laplace smoothing!
    Parameters:
    -----------
    EmpFrame : pandas EmpFrame
        Dataframe containing one year emperical data
    Delta : integer
        How many years in each block
    type : char string
        The type of model to make, 'Laplace' or 'MLE'
    
    Returns
    -------
    pandas DataFrame containing the model.
    """
    #Resample the dataframe by Delta:
    if Delta > 1:
        EmpFrame = changePeriod(EmpFrame,Delta=Delta, MovingAverage = MovingAverage)
        
    #Generate empty dataframe of the same size as the one give:
    ModelFrame = pd.DataFrame(columns=EmpFrame.columns, index=EmpFrame.index)
    DFlen = EmpFrame.shape[1]
    if  DFlen > 63: #Categories
        Osize = len(c2n.offsubcat)
        Psize = len(c2n.punsubcat)+1
    else: #SubCategories
        Osize = len(c2n.offcat)
        Psize = len(c2n.puncat)+1
    Ocount = np.zeros(Osize)
    Pcount = np.zeros(Psize)
    
    #Loop through rows, in each row we
        #count offences and punishments, creating Ocount and Pcount
        #Generate estimate from counts Oprob, Pprob
        #Generate the matrix of probabilities, then unwrap matrix into row...
    for row,rowname in enumerate(EmpFrame.index):
         #Offences
        for o in range(0,Osize):
            Ocount[o] = EmpFrame.iloc[row,o*Psize:(o+1)*Psize].sum()
        #Punishments
        for p in range(0,Psize):
            Pcount[p] = EmpFrame.iloc[row, list(range(p,DFlen,Psize))].sum()
        if type == 'Laplace':
            ##Generate probability estimate via Laplace
            if row == 0:
                #Alpha =1 
                Oprob = (Ocount+1)/(Ocount.sum() + Ocount.shape[0])
                Pprob = (Pcount+1)/(Pcount.sum() + Pcount.shape[0])
                OprobPrevious = Oprob*1
                PprobPrevious = Pprob*1
            else:
                #Alpha = *probPrevious
                Oprob = (Ocount + OprobPrevious) / (Ocount.sum()+1)
                Pprob = (Pcount + PprobPrevious) / (Pcount.sum()+1)
                OprobPrevious = Oprob*1
                PprobPrevious = Pprob*1
        elif type == 'MLE':
            ##Generate probability estimate via MLE
            Oprob = (Ocount)/(Ocount.sum())
            Pprob = (Pcount)/(Pcount.sum())
        else:
            print('Unknown type: {}'.format(type))
            return
        #Generate probability matrix by multiplying the (matrified)transpose of the Oprob with the Pprob
        Pmatrix = (Oprob[np.newaxis].T)*Pprob
        #Unwrap the Pmatrix to rows of the ModelFrame:
        ModelFrame.iloc[row] = np.asarray(Pmatrix).reshape(-1)
        
    return ModelFrame.fillna(0)


#For analysing the data and models
    
def loglikilyhood(EmpiricalFrame, ModelFrame):
    """Calculate the log likilyhood row by row """
    #print ((EmpiricalFrame*np.log(ModelFrame)).sum(axis=1).sum())
    return (EmpiricalFrame*np.log(ModelFrame)).sum(axis=1).sum()

def conditionPunishmentEstimatesOnOffence(CatMod):
    """ Fix the 'relative' estimates for punishment distributions
    """
    #Check whether this is Categories or Subcategories
    if  CatMod.shape[1] > 63:
        #Is subcategories
        p = 28
        o = 57
    elif CatMod.shape[1] is 18:
        p = 2
        o = 9
    else:
        p = 7
        o = 9    
    
    CatModCond = CatMod.mul(1)
    for c in range(0,o):
        CatModCond.iloc[:,c*p:(c+1)*p] = CatMod.iloc[:,c*p:(c+1)*p].div(
            CatMod.iloc[:,c*p:(c+1)*p].sum(axis=1),axis=0).fillna(0)
    return CatModCond

def conditionOffenceEstimatesOnPunishment(CatMod):
    """ Fix the 'relative' estimates for punishment distributions
    """
    #Check whether this is Categories or Subcategories
    ModLen = CatMod.shape[1]
    if   ModLen > 63:
        #Is subcategories
        p = 28
        o = 56
    else:
        p = 7
        o = 9 
    
    CatModCond = CatMod.mul(1)
    for c in range(0,p):
        CatModCond.iloc[:,list(range(c,ModLen,p))] = CatMod.iloc[:,list(range(c,ModLen,p))].div(
            CatMod.iloc[:,list(range(c,ModLen,p))].sum(axis=1),axis=0)
    return CatModCond

def conditionModFile(Filename):
    """
    Find conditional probabilities for offences and punishments and save in their respective files.
    """
    CatMod = loadMod(Filename)
    conditionOffenceEstimatesOnPunishment(CatMod).to_csv(Filename.replace('Dependent' , 'DependentCondOffOnPun'),
                                                         na_rep = 0)
    conditionPunishmentEstimatesOnOffence(CatMod).to_csv(Filename.replace('Dependent' , 'DependentCondPunOnOff'),
                                                         na_rep =0)

def conditionalGenderDifference():
    """
    Generate conditional gender differences.
    """
    for Cond in ['OffOnPun','PunOnOff']:
        for Cat in ['','SubCat']:
            (loadMod('Data/Models/OBO'+Cat+'ModelDependentCond'+Cond+'Male2yr.csv')
                -loadMod('Data/Models/OBO'+Cat+'ModelDependentCond'+Cond+'Female2yr.csv')
                ).to_csv('Data/Models/OBO'+Cat+'ModelDependentCond'+Cond+'Difference2yr.csv')

def conditionDeathOrNot(ModPath='Data/Models/',Prefix='OBOModelDependent',OutputPath='Data/Models/'):
    for f in [1,2,3,4,5,10,50,100,234]:
        conditionModFile(ModPath + Prefix + 'MLECatDeathOrNotMovingAverageNoGender' + str(f) + 'yr.csv')
        conditionModFile(ModPath + Prefix + 'LaplaceCatDeathOrNotMovingAverageNoGender' + str(f) + 'yr.csv')

def generateTVD(DataFrame, OutputDir='Data/TVD/',Gender = 'NoGender', Suffix=''):
    """Generate CSV's containing the cumulate variational difference and total variational difference.
    
    Parameters:
    -----------
    ModFrame : pandas DataFrame
        A DataFrame of estimates
    """
    DFlen = ModFrame.shape[1]
    if  DFlen > 63:
        DFType = 'SubCat'
    else:
        DFType = 'Cat'
    Variation = pd.ModFrame(index=ModFrame.index)
 
    #Calculate estimate from empirical frequencies using Laplace smoothing with alpha=1
    #ModFrame = (ModFrame+1).div(ModFrame.sum(axis=1).values+ModFrame.shape[1],axis=0)
    #SubCatFrame = (SubCatFrame+1).div(SubCatFrame.sum(axis=1).values+SubCatFrame.shape[1],axis=0)
    
    #Calculate consecutive total variation distance and TV per year
    Variation['CumulativeTVD'] = 0.5*ModFrame.diff().abs().sum(axis=1)
    for row in ModFrame.index:
        Variation[row]=0.5*(ModFrame-ModFrame.loc[row]).abs().sum(axis=1)
    Variation['TVsum'] = Variation.iloc[:,1:].sum().values
    Variation['TVsumScaled'] = Variation['TVsum']/Variation['TVsum'].max()
    
    ##Write out files
    #ModFrame.to_csv(OutputDir+FileName+'Estimate'+'.csv',na_rep=0)
    Variation.to_csv(OutputDir+'OBOVariation'+Suffix+Gender+'.csv',na_rep=0)

def generateConsecutiveTVD(ModFrame):
    """As above, but takes in a Model Frame and returns a consecutive TVD series"""
    return 0.5*ModFrame.diff().abs().sum(axis=1)
    
def generateModelFiles(EmpFrame, Prefix='Cat', Type='Dependent', MovingAverage=False, Path='Data/Models/', Suffix='NoGender'):
    """Generate model csv files to the path given for a list of timeframes.
    """
    for f in [1,2,3,4,5,10,50,100,240]:
    #for f in [1,2,5,10,50]:
        if Type == 'Dependent':
            generateDependentModelLaplace(EmpFrame,Delta=f, MovingAverage = MovingAverage).to_csv(Path + 'OBOModelDependent' + Prefix + Suffix + str(f) + 'yr.csv')
            #generateDependentModelMLE(EmpFrame,Delta=f, MovingAverage = MovingAverage).to_csv(Path + 'OBOModelDependentMLE' + Prefix + Suffix + str(f) + 'yr.csv')
        elif Type == 'Independent':
            generateIndependentModel(EmpFrame,Delta=f, MovingAverage = MovingAverage,type='Laplace').to_csv(Path + 'OBOModelIndependent' + Prefix + Suffix + str(f) + 'yr.csv')
            #generateIndependentModel(EmpFrame,Delta=f, MovingAverage = MovingAverage,type='MLE').to_csv(Path + 'OBOModelIndependentMLE' + Prefix + Suffix + str(f) + 'yr.csv')
        elif Type == 'Conditional':
            print('Do this manually for now, and only on Categories')
        elif Type == 'All':
            generateDependentModelLaplace(EmpFrame,Delta=f, MovingAverage = MovingAverage).to_csv(Path + 'OBOModelDependent' + Prefix + Suffix + str(f) + 'yr.csv')
            #generateDependentModelMLE(EmpFrame,Delta=f, MovingAverage = MovingAverage).to_csv(Path + 'OBOModelDependentMLE' + Prefix + Suffix + str(f) + 'yr.csv')
            generateIndependentModel(EmpFrame,Delta=f, MovingAverage = MovingAverage,type='Laplace').to_csv(Path + 'OBOModelIndependent' + Prefix + Suffix + str(f) + 'yr.csv')
            #generateIndependentModel(EmpFrame,Delta=f, MovingAverage = MovingAverage,type='MLE').to_csv(Path + 'OBOModelIndependentMLE' + Prefix + Suffix + str(f) + 'yr.csv')
        else:
            print('Model type "{}" was not recognised'.format(Type))

def generateAllGenderModels(RawPath='Data/Raw/', OutputPath = 'Data/Models/'):
    """Generate all the models possible from raw data"""
    Gender = ['NoGender', 'Female', 'Male']
    CatEmps = [loadCatEmp(RawPath + 'OBOextract' + g + '1yr.csv') for g in Gender]
    SubCatEmps = [loadSubCatEmp(RawPath + 'OBOextract' + g + '1yr.csv') for g in Gender]
    
    #Categories
    for g, gender in enumerate(Gender):
        generateModelFiles(CatEmps[g], Prefix='Cat', Type='All', Suffix=gender)
    #SubCategories
    for g, gender in enumerate(Gender):
        generateModelFiles(SubCatEmps[g], Prefix='Subcat', Type='All', Suffix=gender)
        
def generateAICstandardModels(RawPath='Data/Raw/', ModelPath='Data/Models/', OutputPath='Data/AIC/AICscore.csv', Prefix='Cat', Suffix='NoGender', MovingAverage = False, ReturnFrame=False):
    """Create a DataFrame containing AIC scores of all standard models. Only NoGender at this stage.
    
    Parameters
    ----------
    RawPath : char string
        Path with trailing slash to the raw data files
    ModelPath: char string
        Path in which model files are to be found
    OutputPath : (relative) path in which to write out the resulting DataFrame.
    
    Returns:
    pandas DataFrame
    """
    AICFrame = pd.DataFrame(columns = ['FileName','k','ll','AIC'])
    CatEmp = loadCatEmp(RawPath + 'OBOextractNoGender1yr.csv')
    
    Deltas = [1,2,3,4,5,10,50,100,240]
    regex = re.compile(r'\d+')
    regex
    #Generate all the refactorings:
    CatEmps = [ changePeriod(CatEmp,Delta, MovingAverage = MovingAverage) for Delta in Deltas]
    #Create empty DataFrame
    AICFrame = pd.DataFrame(columns = ['Model','k','ll','AIC'])
    #For all files in the path ModelPath with NoGender and Cat in their name
    for idx,file in enumerate(glob.glob(ModelPath + '*'+ Prefix + '*' + Suffix +'*')):
        print('Finding AIC score for model from file {}'.format(file))
        fileYearDelta = int(regex.search(file).group(0))
        DeltasIndex = Deltas.index(fileYearDelta)
        try:
            Mod = loadMod(file)
            filename = pathlib.PurePath(file).stem
            if 'Dependent' in file:
                k = (9*7-1)*CatEmps[DeltasIndex].shape[0]
            elif 'Independent' in file:
                k = (9+7-2)*CatEmps[DeltasIndex].shape[0]
            elif 'Partition' in file:
                print('Not done yet!')
            ll = loglikilyhood(CatEmps[DeltasIndex], Mod)
            AIC = 2*k - 2*ll
            AICFrame.loc[idx] = [ filename, k, ll, AIC]
            #print('{}, years: {}, model index {}, k {}, log-likelihood {}, AIC {}'.format(file,fileYearDelta,DeltasIndex,k,ll, AIC))
            #print('{} loaded'.format(file))
        except:
            print('{} failed to load'.format(file))
    AICFrame.sort('AIC', inplace=True)
    
    if ReturnFrame:
        return AICFrame
    else:
        AICFrame.to_csv(OutputPath)

    
def generateLogLikilyhood(EmpiricalFramePath, ModelDir='Data/Models/'):
    """Generate a list of log likilyhoods for the given frame of emperical frequencies."""
    #Get category and subcategory frames
    Emps = [ loadCatEmp(EmpiricalFramePath),
            loadSubCatEmp(EmpiricalFramePath)]
    Prefixes = ['Cat','SubCat']
    ModelTypes = ['Independent','Dependent']

    #An array to store it all in
    AnArray =[]

    #Generating these file names to compare: OBOModelDependentCat100yr.csv
    #Loop through stuff

    #Category, Subcategory
    for index, frame in enumerate(Emps):
        #Dependent, Independent
        for types in ModelTypes:
            for f in [1,2,3,4,5,10,50,100,240]:
                ModelFrame = pd.DataFrame.from_csv(ModelDir + 'OBOModel' + types + Prefixes[index] + str(f) + 'yr.csv',index_col='Date')
                loglikely = loglikilyhood(frame,ModelFrame)
                print('llh of {} for model type {} and year delta {} is: {}'.format(Prefixes[index],types,f,loglikely))
                AnArray.append(loglikely)
    return AnArray

def generateBootStrapEstimateConfidenceIntervals(EmpFrame, N=1000, Delta=2, Verbose=False):
    """Generate the low and high confidence interval from bootsrtap data using Laplace smoothed model on Delta years.
    
    Parameter:
    ----------
    EmpFrame : pandas DataFrame, DataFrame of empirical frequencies, assumed to be one year empirical data as generated by OBOapiExtraction.
    N : int, number of bootstrapped samples to produce.
    Delta : int, number of years on which to resample data.
    
    This function works by creating a pandas Panel (an array of DataFrames) to store the N bootstrap DataFrames. The panel is built up row wise as folows:
     For every row in the EmpFrame:
        * Turn the event counts (event 0 happens 3 times, event 1 happens twice) into a list of single events as EventList=[0,0,0,1,1,2...] which has a length (L=len(EventList)) equal to the number of events in that row/year.
        * From this EventList we randomly sample by producing a vector the same length as L with integer entries from 0 to L, which we turn into events (0,1) by using this vector as an index on the EventList.
        * We then count the number of events, recovering the original row length of the EmpFrame. We must check that the frame is of the same length adding 0's to fil the possibly empty tailing values.
        * NOTE: We do this not on a vector but on a matrix such that for each row we resample N times at once (so produce a random matrix NxL)
    Now that we have a Panel of N bootstraped Emp frames we apply the change period and Laplace smoothing to each DataFrame in the panel, then find the quantile/percentile across all panels, recoving a DataFrame the same size as the original EmpFrame.
    """
    #Define a verbose printing:
    verboseprint = print if Verbose else lambda *a, **k: None
    
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
        verboseprint('Generating bootstrap data for row {}'.format(row))
        
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
    verboseprint('Reampling data and creating models in Panels')
    BootstrapPanel = BootstrapPanel.apply(lambda x: generateDependentModelLaplace(x,Delta=Delta), axis=[1,2])
     
    #Find upper and lower percentiles and store them in respective DF's
    
    verboseprint('Finding lower and upper percentiles')
    ConfLowFrame  = BootstrapPanel.apply(lambda x: x.quantile(ConfLow),axis=0)
    ConfHighFrame  = BootstrapPanel.apply(lambda x: x.quantile(ConfHigh),axis=0)
    
    return ConfLowFrame, ConfHighFrame
    #return BootstrapPanel
    #return EmpCountList, RandomSampleFrame
    

#def generateAllBootstraps(EmpFrame,Delta=3):
    #EmpFrameChangePeriod = changePeriod(EmpFrame,Delta=Delta)
    
    #for opptype in ['All','Offence','Punishmnet']:
        
    
    
    
#Used to automatically define date ranges
# From: http://stackoverflow.com/a/10619525/398969
def cross(series, cross=0, direction='cross', output='index'):
    """
    Given a Series returns all the index values where the data values equal 
    the 'cross' value. 

    Direction can be 'rising' (for rising edge), 'falling' (for only falling 
    edge), or 'cross' for both edges
    
    Ooutput can be 'index' for the series index, 'indexValue' for the series index value, 'value' for the 'y' value at, 'interpolated', the interpolated value between the crossing and the next.
    """
    # Find if values are above or bellow yvalue crossing:
    above=series.values > cross
    below=np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    x1 = series.index.values[idxs]
    x2 = series.index.values[idxs+1]
    y1 = series.values[idxs]
    y2 = series.values[idxs+1]
    x_crossings = (cross-y1)*(x2-x1)/(y2-y1) + x1

    #return x_crossings
    if output == 'index':
        return idxs
    elif output == 'indexValue':
        return x1
    elif output == 'value':
        return y1
    elif output == 'interpolated':
        return x_crossings

#Generate a model with a non regular time period as found by cross (i.e BlockList = cross(...))
def resampleFrequenciesOnList(EmpFrame,BlockList):
    """Resample the EmpFrame (assumed to be OBO DataFrame on full period) into blocks starting at the index'es in the BlockList."""
    #Get the length of the EmpFrame and Blocklist
    DFlon = EmpFrame.shape[0]
    BLlen = len(BlockList)
    
    #if BlockList is 0:
        #return 0
    #Check that the list starts at 0, advance index
    if BlockList[0] is not 0:
        BlockList = np.insert(BlockList, 0, 0).tolist()
    if BlockList[-1] is not (DFlon-1):
        BlockList = np.append(BlockList, (DFlon-1)).tolist()
    DateGroup = [ EmpFrame.index[IdX] for id,IdX in enumerate(BlockList[:-1]) for repeat in [0]*(BlockList[id+1]-BlockList[id]) ]
    #Add an extra one one the end as inclusive of 1913
    DateGroup = np.append(DateGroup,DateGroup[-1]).tolist()
    return EmpFrame.groupby(DateGroup,axis=0).sum()

    
def AICOnBlocksFromCumulativeTVDcrossing(EmpFrame,consecutiveTVD, crossing):
    """Generate a dependent laplace smoothed model using a Blocklist for date breakpoints.
    
    Parameters
    ----------
    EmpFrame : pandas DataFrame, empirical data
    BlockList : list or numpy array, indexes of dates in the CatEmp frame to divide teh date range up into
    
    Returns
    -------
    ModFrame : pandas DataFrame, model data
    """
    #numList=[len(oboM.cross(ConsecutiveTVD,direction='rising',cross=Cross)) for Cross in  np.linspace(0.01,0.3,num=300)  ]
    BlockList = cross(consecutiveTVD, direction='cross', cross=crossing, output='index')
    ResampledEmpFrame = resampleFrequenciesOnList(EmpFrame,BlockList)
    ResampledModFrame = generateDependentModelLaplace(ResampledEmpFrame)
    ll = loglikilyhood(ResampledEmpFrame,ResampledModFrame)
    k = ResampledEmpFrame.shape[0]*(9-1)*(7-1)
    AIC = 2*k - 2*ll
    return print(k,ll,AIC)
