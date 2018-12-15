#!/usr/bin/python


from utils import *
from SyntheticEventDataGenerator import *
import pandas as pd
import sys

def runDataGenerator(filename, numhmmstatesactivity, numhmmstatessensors, timeinterval, timestampmodeltype):

    timeinterval = 6
    # This SyntheticEventGenerator object is in charge of generating acitivity and sensor event sequences
    syndatagenerator = SyntheticEventGenerator(filename, numhmmstatesactivity, numhmmstatessensors, timeinterval, timestampmodeltype)
    syntheticsensoreventdataframe = syndatagenerator.run()

    #assemble the final synthetic data and print
    for i in range(1, len(syntheticsensoreventdataframe)):
        row = syntheticsensoreventdataframe.loc[i]
        if(not pd.isnull(row["timestamp"])):
            print(str(row["timestamp"]) + "," + row["sensor_name"] + "," + row["sensor_state"] + "," + row["activity_name"])



#"/Users/jess/Desktop/HHdata/converted/Translated/DateRange/hh111week1"
if __name__ == "__main__":
    args = sys.argv
    filename = args[1]
    numhmmstatesactivity = int(args[2])
    numhmmstatessensors = int(args[3])
    timeinterval = int(args[4])
    timestampmodeltype = int(args[5])

    runDataGenerator(filename, numhmmstatesactivity, numhmmstatessensors, timeinterval, timestampmodeltype)







