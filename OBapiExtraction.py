#!/usr/bin/env python3
"""
    Download the complete OBO data per day via the API.
    
    For each gender (All, Male, Femal, Other), get daily histograms for
    charges and subcharges to not guilty, punishments, and subpunishments.
    
    We will use urllib3 for the downloading due to its connection pool
    allowing the socket to stay open, and save into a Pandas object to be
    exported to CSV periodically.
    
    Eventually we will have to figure out where to continue from should
    the process be interrupted.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib3

import CategoryToNumberAssignment as c2n

def initialiseDataFrame():
    """Create an empty dataframe with nothing but categories"""
    import pandas as pd
    return pd.DataFrame(columns=c2n.generateCategories())

def initialiseEmptyArray():
    """Initialise an empty array the same length as the Categories"""
    return [0]*len(c2n.generateCategories())

def initialiseEmptyCatArray():
    """Initialise empty array for categories and subcategries.
    
    We need a fixed length empty array the length of all punishment
    categories and subcategories + 1 for not guilty
    """
    import CategoryToNumberAssignment as c2n
    return [0]*(1+len(c2n.puncat))

def initialiseEmptySubCatArray():
    """Initialise empty array for categories and subcategries. """
    import CategoryToNumberAssignment as c2n
    return [0]*(1+len(c2n.punsubcat))   

def generateURLCategoryNotGuilty(Date, OffenceCategory,Gender):
    """Generate an OBOApi url for insertion to an http 'GET'"""
    if Gender is 'None':
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=date_' + Date+'&term1=offcat_' + OffenceCategory+'&term1=vercat_notGuilty&breakdown=cat&count=1'
    else:
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=date_' + Date+'&term1=offcat_' + OffenceCategory+'&term1=defgen_' + Gender+'&term2=vercat_notGuilty&breakdown=offcat&count=1'
    
def generateURLCategory(Date, OffenceCategory,Gender):
    """Generate an OBOApi url for insertion to an http 'GET'"""
    if Gender is 'None':
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=date_' + Date+'&term1=offcat_' + OffenceCategory+'&breakdown=puncat&count=1'
    else:
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=date_' + Date+'&term1=offcat_' + OffenceCategory+'&term1=defgen_' + Gender+'&breakdown=offcat&count=1'

def generateURLCategoryNotGuiltyRange(StartDate, EndDate,Gender='None'):
    """Generate an OBOApi url for insertion to an http 'GET'"""
    if Gender is 'None':
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=vercat_notGuilty&breakdown=offcat&count=1'
    else:
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=vercat_notGuilty&term3=defgen_' + Gender+'&breakdown=offcat&count=1'
    
def generateURLSubCategoryNotGuiltyRange(StartDate, EndDate,Gender='None'):
    """Generate an OBOApi url for insertion to an http 'GET'"""
    if Gender is 'None':
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=vercat_notGuilty&breakdown=offsubcat&count=1'
    else:
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=vercat_notGuilty&term3=defgen_' + Gender+'&breakdown=offsubcat&count=1'
    
def generateURLCategoryRange(StartDate, EndDate, OffenceCategory,Gender='None'):
    """Generate an OBOApi url for insertion to an http 'GET'"""
    if Gender is 'None':
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=offcat_' + OffenceCategory + '&breakdown=puncat&count=1'
    else:
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=offcat_' + OffenceCategory + '&term3=defgen_' + Gender + '&breakdown=puncat&count=1'
    
def generateURLSubCategoryRange(StartDate, EndDate, OffenceCategory,Gender='None'):
    """Generate an OBOApi url for insertion to an http 'GET'"""
    if Gender is 'None':
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=offsubcat_' + OffenceCategory + '&breakdown=punsubcat&count=1'
    else:
        return 'http://www.oldbaileyonline.org/obapi/ob?term0=fromdate_' + StartDate+'&term1=todate_' + EndDate+'&term2=offsubcat_' + OffenceCategory + '&term3=defgen_' + Gender + '&breakdown=punsubcat&count=1'

def URLtoJSON(URL, httpPool):
    import json
    return json.loads( httpPool.request('GET', URL).data.decode('utf8') )

def generateRow(Date, Gender,Sockets=1):
    """Generate one row of data for a given date and defendent gender
    
    We are not atomising the code to generalise URL gets such that
    the socket opened by urllib3 can stay open.
    """
    import CategoryToNumberAssignment as c2n
    import urllib3
    import json
    http=urllib3.PoolManager(maxsize=Sockets)

    Row = [Date]
    
    #Associate Offence Categories with Punishment Categories
    for Category in c2n.offcat:
        _TempCategories = initialiseEmptyCatArray()
        
        # First get the Category -> Not guilty to JSON
        _Json = json.loads( http.request('GET', generateURLCategoryNotGuilty(Date,Category,Gender)).data.decode('utf8') )
        #Not guilty totals
        for Totals in _Json['breakdown']:
            _TempCategories[0] += Totals['total']
            
        # Now pull all values out of JSON and put into empty array
        #Get the Json data
        _Json = json.loads( http.request('GET', generateURLCategory(Date,Category,Gender)).data.decode('utf8') )
        #Find the punishment totals and place in the correct position in the array 
        # adding 1 place for the Not guilty
        for Totals in _Json['breakdown']:
            _TempCategories[c2n.puncat.index(Totals['term'])+1]=Totals['total']
        
        # Append _Temps to the Rows
        Row = Row + _TempCategories
        
    #Associate Offence Subcategories with Punishment Subategories
    for Category in c2n.offsubcat:
        _TempCategories = initialiseEmptySubCatArray()
        
        # First get the Category -> Not guilty to JSON
        _Json = json.loads( http.request('GET', generateURLCategoryNotGuilty(Date,Category,Gender)).data.decode('utf8'))
        #Not guilty totals
        for Totals in _Json['breakdown']:
            _TempCategories[0] += Totals['total']
            
        # Now pull all values out of JSON and put into empty array
        #Get the Json data
        _Json = json.loads( http.request('GET', generateURLCategory(Date,Category,Gender)).data.decode('utf8') )
        #Find the punishment totals and place in the correct position in the array 
        # adding 1 place for the Not guilty
        for Totals in _Json['breakdown']:
            _TempCategories[c2n.punsubcat.index(Totals['term'])+1]=Totals['total']
                
        # Append _Temps to the Rows
        Row = Row + _TempCategories
        
    return Row

def generateRowRange(StartDate, EndDate, Gender='None', http='None',Sockets=2):
    """Generate one row of data for a given date and defendent gender
    
    We are not atomising the code to generalise URL gets such that
    the socket opened by urllib3 can stay open. This may be messy 
    looking...
    """
    import CategoryToNumberAssignment as c2n
    import urllib3
    import json
    if http is 'None':
        http=urllib3.PoolManager(maxsize=Sockets)
        
    sStartDate = str(StartDate)
    sEndDate = str(EndDate)
    
    _Columns = c2n.generateCategories()
    
    #Generate empty columns
    Row = [sStartDate]
    
    """Get not breakdown of punishments by category and subcategory for the period"""
    
    # Categories
    for Category in c2n.offcat:
        _TempCategories = initialiseEmptyCatArray()
        
        #Get the Json data
        #print("Generating URL: {}".format(generateURLCategoryRange(sStartDate,sEndDate,Category, Gender)))
        _Json = URLtoJSON(generateURLCategoryRange(sStartDate,sEndDate,Category, Gender),http)
        
        #Find the punishment totals and place in the correct position in the array 
        # adding 1 place for the Not guilty
        for Totals in _Json['breakdown']:
            _TempCategories[c2n.puncat.index(Totals['term'])+1]=Totals['total']
        
        # Append _Temps to the Rows
        Row = Row + _TempCategories
        
    #Associate Offence Subcategories with Punishment Subategories
    for Category in c2n.offsubcat:
        _TempCategories = initialiseEmptySubCatArray()
        
        #Get the Json data
        _Json = URLtoJSON(generateURLSubCategoryRange(sStartDate,sEndDate,Category, Gender),http)
        #Find the punishment totals and place in the correct position in the array 
        # adding 1 place for the Not guilty
        for Totals in _Json['breakdown']:
            _TempCategories[c2n.punsubcat.index(Totals['term'])+1]=Totals['total']
        
        # Append _Temps to the Rows
        Row = Row + _TempCategories
    
    """Get not guilties by category and subcategory for the period"""
    #Get not guilties:
    _JsonNotGuiltyCat = URLtoJSON(generateURLCategoryNotGuiltyRange(sStartDate,sEndDate,Gender),http)
    _JsonNotGuiltySubCat = URLtoJSON(generateURLSubCategoryNotGuiltyRange(sStartDate,sEndDate,Gender),http)
    
    # Place values associated with locations in Row:
    for Totals in _JsonNotGuiltyCat['breakdown']:
        Row[_Columns.index(c2n.upcaseFirstLetter(Totals['term'])+'NotGuilty')] = Totals['total']
        
    for Totals in _JsonNotGuiltySubCat['breakdown']:
        Row[_Columns.index(c2n.upcaseFirstLetter(Totals['term'])+'NotGuilty')] = Totals['total']
        
    return Row


def generateDataFrames(Date_List, Gender):
    """Generate the dataframes from a gender and datelist"""
    import pandas as pd
    
    DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    for dAte in Date_List:
        print('Generating row for {}'.format(dAte))
        DummyFrame.loc[len(DummyFrame)] = generateRow(dAte,Gender)
        
    return DummyFrame


def generateDataFramesRange(StartDate, EndDate, DateStep, Gender='None'):
    """Generate the dataframes from a gender and datelist"""
    import pandas as pd
    import urllib3
    retry = urllib3.util.Retry(total=1000, read=200, connect=200, backoff_factor=0.5)
    timeout = urllib3.util.Timeout(connect=2.0, read=4.0)
    http=urllib3.PoolManager(retry=retry, timeout=timeout, maxsize=10)
    
    DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    for _startDate in range(StartDate,EndDate+1,DateStep):
        _endDate= _startDate+DateStep-1
        print('Generating row for {} to {}:'.format(_startDate,_endDate))
        DummyFrame.loc[len(DummyFrame)] = generateRowRange(_startDate, _endDate, Gender)
        
    return DummyFrame


def generateDataFramesParallel(Date_List, Gender):
    """Generate the dataframes from a gender and datelist"""
    import workerpool
    import json
    import pandas as pd
    
    Gender = 'None'
    DummyFrame = pd.DataFrame(columns=c2n.generateCategories())

    NUM_SOCKETS = 3
    NUM_WORKERS = 5

    # We want a few more workers than sockets so that they have extra
    # time to parse things and such.
    workers = workerpool.WorkerPool(size=NUM_WORKERS)

    class MyJob(workerpool.Job):
        def __init__(self, dAte, Gender):
            self.dAte = dAte
            self.Gender = Gender

        def run(self):
            print('Generating row for {}'.format(self.dAte))
            DummyFrame.loc[len(DummyFrame)] = generateRow(self.dAte,self.Gender,NUM_SOCKETS)
    
    for daTe in Date_List:
        workers.put(MyJob(daTe,Gender))

    # Send shutdown jobs to all threads, and wait until all the jobs have been completed
    # (If you don't do this, the script might hang due to a rogue undead thread.)
    workers.shutdown()
    workers.wait()
    
    return DummyFrame
        
def generateDataFrameInChunks(Date_List,Gender,ChunkSize=10, File='OBOextract',  Path='Data/Raw/', Overwrite = 1, Start=0, Stop=0):
    """ Generate DataFrames, but save to csv every so often"""
    import pandas as pd
    
    if Stop <= Start:
        Stop=len(Date_List)
    
    DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    for i in range(Start,Stop,ChunkSize):
        for dAte in Date_List[i:i+ChunkSize-1]:
            print('Generating row for date {}, and gender {}'.format(dAte,Gender))
            DummyFrame.loc[len(DummyFrame)] = generateRow(dAte,Gender)
        DummyFrame.to_csv(File+'.csv')
        
    return DummyFrame


def generateDataFramesRangeSaving(StartDate, EndDate, DateStep, Gender='None', SaveEvery=1, File='OBOextract', Path='Data/Raw/', Overwrite = 1):
    """Generate the dataframes from a gender and datelist"""
    import os
    import pandas as pd
    import urllib3
    retry = urllib3.util.Retry(total=100, read=100, connect=100, backoff_factor=1)
    timeout = urllib3.util.Timeout(connect=4.0, read=8.0)
    http=urllib3.PoolManager(retry=retry, timeout=timeout, maxsize=5)

    _Iteration = 0
    
    DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    if Overwrite is 0:
        # Does the file exist
        if os.path.isfile(Path + File+'.csv'):
            try:
                DummyFrame = pd.DataFrame.from_csv(Path + File+'.csv')
                _TempStartDate = int(DummyFrame.tail(1).values[0,0] + DateStep)
                if _TempStartDate >= StartDate:
                    StartDate = _TempStartDate + DateStep
            except:
                DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    for _startDate in range(StartDate,EndDate+1,DateStep):
        _Iteration += 1
        _endDate= _startDate+DateStep-1
        print('Generating row for {} to {}, Gender: {}'.format(_startDate,_endDate,Gender))
        DummyFrame.loc[len(DummyFrame)] = generateRowRange(_startDate, _endDate, Gender)
        if _Iteration >= SaveEvery:
            _Iteration = 0
            DummyFrame.to_csv(Path + File+'.csv')
            
    return DummyFrame

def generateDataFramesRollingRangeSaving(StartDate, EndDate, DateStep, Delta=1000, Gender='None', SaveEvery=1, File='OBOextract',  Path='Data/Raw/', Overwrite = 1):
    """Generate the dataframes from a gender and datelist
	For dates from StartDate to EndDate stepped through by DateStep,
	data is extracted for the date to date+Delta-1.
	The data is saved in a Pandas DataFrame and cumulativly updated, and saved every SaveEvery retrievals to the file File.
	If Overwrite=1 then the process will attempt to restart from interruption based on the xistingesaeved DataFdramed.dd"""
    import os
    import pandas as pd
    import urllib3
    retry = urllib3.util.Retry(total=100, read=100, connect=100, backoff_factor=1)
    timeout = urllib3.util.Timeout(connect=4.0, read=8.0)
    http=urllib3.PoolManager(retry=retry, timeout=timeout, maxsize=5)

    _Iteration = 0
    
    DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    if Overwrite is 0:
        # Does the file exist
        if os.path.isfile(Path + File+'.csv'):
            try:
                DummyFrame = pd.DataFrame.from_csv(Path + File+'.csv')
                _TempStartDate = int(DummyFrame.tail(1).values[0,0] + DateStep)
                if _TempStartDate >= StartDate:
                    StartDate = _TempStartDate + DateStep
            except:
                DummyFrame = pd.DataFrame(columns=c2n.generateCategories())
    
    for _startDate in range(StartDate,EndDate+1,DateStep):
        _Iteration += 1
        _endDate= _startDate+Delta-1
        print('Generating row for {} to {}, Gender: {}'.format(_startDate,_endDate,Gender))
        DummyFrame.loc[len(DummyFrame)] = generateRowRange(_startDate, _endDate, Gender)
        if _Iteration >= SaveEvery:
            _Iteration = 0
            DummyFrame.to_csv(Path + File+'.csv')
            
    return DummyFrame

def GetEmAll():
    """ Downlad as much as is possible of the OBOarchive"""
    generateDataFramesRangeSaving(16740000, 19130000, 10000, 'None', SaveEvery=1, File='OBOextractNoGender1yr', Overwrite = 0)
    generateDataFramesRangeSaving(16740000, 19130000, 10000, 'male', SaveEvery=1, File='OBOextractMale1yr', Overwrite = 0)
    generateDataFramesRangeSaving(16740000, 19130000, 10000, 'female', SaveEvery=1, File='OBOextractFemale1yr', Overwrite = 0)
    generateDataFramesRangeSaving(16740000, 19130000, 10000, 'indeterminate', SaveEvery=1, File='OBOextractIndeterminate1yr', Overwrite = 0)

"""Run the thing if we execute this script"""
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--getEmAll", action='store_true', help="Get all the NoGender metadata from the OBO")
args = parser.parse_args()
if args.getEmAll is True:
    GetEmAll()
