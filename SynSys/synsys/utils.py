import pandas as pd
import csv


def getDataFrame(filename):
    headers = ["timestamp", "sensor_name", "sensor_state", "activity_name"]
    shcategoricaldata = pd.read_csv(filename, header=None, names=headers)
    shcategoricaldata['timestamp'] = pd.to_datetime(shcategoricaldata['timestamp'])
    shcategoricaldata.index = shcategoricaldata['timestamp']
    del shcategoricaldata['timestamp']
    return shcategoricaldata

def getDataFrameSpaces(filename):
    headers = ["date", "timestamp", "sensor_name", "sensor_state", "activity_name"]
    shcategoricaldata = pd.read_csv(filename, header=None, names=headers, delim_whitespace=True)
    shcategoricaldata['timestamp'] = shcategoricaldata["date"] + " " +  shcategoricaldata["timestamp"]
    del shcategoricaldata["date"]
    shcategoricaldata['timestamp'] = pd.to_datetime(shcategoricaldata['timestamp'])
    shcategoricaldata.index = shcategoricaldata['timestamp']
    del shcategoricaldata['timestamp']
    return shcategoricaldata


#return week of data based on date provided
#None if date not present in dataset
def getDataDateRange(datestr1,datestr2, inputfilename, outputfilename):
    data = getDataFrame(inputfilename)
    weekdata = data[datestr1:datestr2]
    #print(weekdata.head())
    weekdata.to_csv(outputfilename, header=None)
    return weekdata








