import pandas as pd
from pomegranate import *
from utils import *
import random as rand
from collections import Counter
import numpy as np
from sklearn import *
import datetime
from statsmodels.discrete.discrete_model import Poisson

class SyntheticEventGenerator():

    #initializes generator data by building a dataframe of the real data
    def __init__(self, datafilename, numhmmstatesactivity, numhmmstatessensors, timeinterval, timestampmodeltype):
        self.timeinterval = timeinterval
        self.datafilename = datafilename
        self.realdataframe = self.generateRealDataFrame(self.datafilename).sort_index()

        self.activitydurationdist = {}

        self.mostlikelyactivities = self.getMostLikelyActivitiesDict(self.realdataframe, timeinterval)

        self.activitynamemapnumtostr = dict(enumerate(self.realdataframe['activity_name'].cat.categories))
        self.activitynamemapstrtonum = {v: k for k, v in self.activitynamemapnumtostr.items()}

        self.sensornamemapnumtostr = dict(enumerate(self.realdataframe['sensor_name'].cat.categories))
        self.sensornamemapstrtonum = {v: k for k, v in self.sensornamemapnumtostr.items()}


        self.sensorstatemapnumtostr = dict(enumerate(self.realdataframe['sensor_state'].cat.categories))
        self.sensorstatemapstrtonum = {v: k for k, v in self.sensorstatemapnumtostr.items()}

        realactivitysequence = self.realdataframe["activity_name"]
        self.realactivitysequence = realactivitysequence.loc[realactivitysequence.shift() != realactivitysequence]

        self.activitynamesunique = self.realdataframe["activity_name"].unique()
        self.sensornamesunique = self.realdataframe["sensor_name"].unique()
        self.sensorstatesunique = self.realdataframe["sensor_state"].unique()

        self.sensorEventNumDistributions = {}

        self.getSensorEventDistributions(self.realdataframe)

        self.sensoreventhmms = {}

        self.numhmmstatesactivity = numhmmstatesactivity
        self.numhmmstatessensors = numhmmstatessensors

        self.timestampmodel = self.getTimestampModel(timestampmodeltype)


        self.fitActivityHMM()
        self.fitSensorEventHMMS()

        self.isinit = True

    def getTimestampModel(self, timestampmodeltype):
        if(timestampmodeltype == 1):
            return linear_model.LinearRegression()
        elif(timestampmodeltype == 2):
            return linear_model.Ridge()
        elif(timestampmodeltype == 3):
            return linear_model.Lasso()
        elif(timestampmodeltype == 4):
            return linear_model.ElasticNet()
        else:
            return linear_model.LinearRegression()

    #Get the number of seconds since midnight
    def getTimeOfDay(self, row):
        currentime = row['Date']
        timeofday = (currentime - currentime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        return timeofday

    #get the in value for the day of the week
    def getDayOfWeek(self, row):
        currentime = row['Date']
        dayofweek = currentime.date().weekday()
        return dayofweek

    #Make a dataframe with the real data
    def generateRealDataFrame(self, datafilename):

        data = getDataFrame(datafilename)

        data["sensor_namestate"] = data["sensor_name"] + "_" + data["sensor_state"]
        data["sensor_namestate"] = pd.Categorical(data["sensor_namestate"])
        data["sensor_namestate_code"] = data.sensor_namestate.cat.codes

        data["activity_name"] = pd.Categorical(data["activity_name"])
        data["sensor_name"] = pd.Categorical(data["sensor_name"])
        data["sensor_state"] = pd.Categorical(data["sensor_state"])


        data["activity_name_code"] = data.activity_name.cat.codes
        data["sensor_name_code"] = data.sensor_name.cat.codes
        data["sensor_state_code"] = data.sensor_state.cat.codes

        data['Date'] = pd.to_datetime(data.index)

        data['NextDate'] = data['Date'].shift(-1)

        data["time_of_day"] = data.apply(self.getTimeOfDay, axis=1)
        data['day_of_week'] = data.apply(self.getDayOfWeek, axis=1)

        data['hidden_state'] = -1

        return data


    #Get the number of sensor events for each type of activity
    def getSensorEventDistributions(self, realDataFrame):

        #set up dictionary of lists
        for activity in self.activitynamesunique:
            self.sensorEventNumDistributions[activity] = []

        currentactivity = realDataFrame.iloc[0]["activity_name"]
        currentactivitysensorcount = 0
        for index, row in realDataFrame.iterrows():

            if(row["activity_name"] != currentactivity):
                self.sensorEventNumDistributions[currentactivity].append(currentactivitysensorcount)
                currentactivity = row["activity_name"]
                currentactivitysensorcount = 0

            currentactivitysensorcount += 1

    #create an HMM model for the activity seqeunce, fit to the real data
    def fitActivityHMM(self):
        realactivitysequencematrix = self.realactivitysequence.as_matrix()
        realactivitysequencematrix = realactivitysequencematrix.reshape(-1, 1)
        model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=self.numhmmstatesactivity, X=realactivitysequencematrix)
        self.activityhmmmodel = model


    #create HMM models for the sensor events for each activity
    def fitSensorEventHMMS(self):

        # Train separate hmm's for each real activity
        for realactivity in self.activitynamesunique:
            rows = self.realdataframe.loc[self.realdataframe['activity_name'] == realactivity]

            statenameseq = rows["sensor_namestate"].tolist()

            sensorevents = np.array(statenameseq).reshape(-1, 1)

            model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=self.numhmmstatessensors, X=sensorevents)

            # add to hmm list
            self.sensoreventhmms[realactivity] = model

    #generate an  activity sequence, start from state most likely to generate activity if activity is not None
    def generateActivitySequence(self, activity, numsamples):
        if(self.isinit):
            if(activity == None):#use default start state
                syntheticactivitysequence, path = self.activityhmmmodel.sample(numsamples, path=True)
                path.pop(0)
                synthetichiddenstatesequence = []
                for state in path:
                    index = self.activityhmmmodel.getStateIndex(state.name)
                    synthetichiddenstatesequence.append(index)

                return syntheticactivitysequence, synthetichiddenstatesequence
            else:
                #use start state most likely for activity
                
                syntheticactivitysequence, path = self.activityhmmmodel.resample(activity, numsamples, path=True)


                synthetichiddenstatesequence = []
                for state in path:
                    index = self.activityhmmmodel.getStateIndex(state.name)
                    synthetichiddenstatesequence.append(index)
                return syntheticactivitysequence, synthetichiddenstatesequence

    #generate sensor events for the given activity seqeunce
    def generateSensorEventSequence(self, activitySequence):

        syntheticsensoreventdataframe = pd.DataFrame()

        synactivities = []
        synsensornames = []
        synsensorstates = []
        synhiddenstates = []

        activitystartmarkerlist = []
        timestamplist = []
        count = 0
        # get synthetic sequences for each synthetic activity
        for synactivity in activitySequence:
            hmmmodel = self.sensoreventhmms[synactivity]
            #pick random number of sensor events from real distributon
            sensoreventdist = self.sensorEventNumDistributions[synactivity]
            randindex = rand.randrange(0,len(sensoreventdist))
            randnumberofevents = sensoreventdist[randindex]

            sensoreventsamples, path = hmmmodel.sample(randnumberofevents, path=True)

            path.pop(0)
            flag = 0

            for sample in sensoreventsamples:
                synactivities.append(synactivity)
                name, state = sample.split("_")
                synsensornames.append(name)
                synsensorstates.append(state)

                timestamplist.append(self.synthetictimestamps[count])


                if(flag == 0):
                    activitystartmarkerlist.append(1)
                    flag = 1
                else:
                    activitystartmarkerlist.append(0)

            for state in path:
                synhiddenstates.append(hmmmodel.getStateIndex(state.name))
            
            count += 1

        syntheticsensoreventdataframe["sensor_name"] = synsensornames
        syntheticsensoreventdataframe["sensor_state"] = synsensorstates
        syntheticsensoreventdataframe["activity_name"] = synactivities
        syntheticsensoreventdataframe["hidden_state"] = synhiddenstates
        syntheticsensoreventdataframe["previous_hidden_state"] = syntheticsensoreventdataframe["hidden_state"].shift(1)
        syntheticsensoreventdataframe["timestamp"] = timestamplist
        syntheticsensoreventdataframe["activitystartmarker"] = activitystartmarkerlist

        
        syntheticsensoreventdataframe["sensor_name_num"] = syntheticsensoreventdataframe["sensor_name"].replace(
            self.sensornamemapstrtonum)
        syntheticsensoreventdataframe["sensor_state_num"] = syntheticsensoreventdataframe["sensor_state"].replace(
            self.sensorstatemapstrtonum)
        syntheticsensoreventdataframe["activity_name_num"] = syntheticsensoreventdataframe["activity_name"].replace(
            self.activitynamemapstrtonum)
        syntheticsensoreventdataframe = syntheticsensoreventdataframe.dropna()
        return syntheticsensoreventdataframe


    # get the most likely activity that occurs in the time window
    # for example get most likely activity in 6 hour windows over n days

    def getMostLikelyActivitiesDict(self, data, timeinterval):

        likleyactivitiesdict = {}

        # I have a data frame with all the info
        # I need to break into chunks of time interval time

        # I can just divide 24 hour day into the chunks and then grab the data that is in each of those time windows
        # go through each time window, select data and get most likely activity, save to dict

        # get each day in the data

        dates = data.Date.dt.strftime('%m-%d-%Y').unique()

        for date in dates:
            parts = date.split("-")
            # get the day, month, year for each day in the data
            month, day, year = parts[0], parts[1], parts[2]
            # set start to start of day at midnight

            start = datetime.datetime(int(year), int(month), int(day), 0, 0, 0)

            # set end to 11:59 PM
            end = datetime.datetime(int(year), int(month), int(day), 23, 59, 0)
            while start <= end:
                startwindowtime = start
                start += datetime.timedelta(hours=timeinterval)
                endwindowtime = start
                # print(startwindowtime, endwindowtime)

                # get the data that is in the start and end window time range
                window = data[startwindowtime:endwindowtime]
                
                mostfreqactivity = "C0"
                if (len(window) != 0):
                    # get the activities from the data chunk
                    # multiple values return if more than one shares max frequency
                    windowactivities = window["activity_name"].mode()

                    # select a random one if there is more than one
                    randommaxactivityindex = rand.randint(0, len(windowactivities) - 1)
                    mostfreqactivity = windowactivities.loc[randommaxactivityindex]

                    # get the actvitiy that occurs the most

                    # save time range and most likely activity to dict

                    # have dict be time range str without dates and then list of activities for each day in the data
                    # likleyactivitiesdict[00:00:00,06:00:00] returns ["Cook", "Eat", "Eat"]

                startwindowtimestr = startwindowtime.strftime('%H:%M:%S')
                endwindtimestr = endwindowtime.strftime('%H:%M:%S')

                try:
                    likleyactivitiesdict[(startwindowtime, endwindowtime)].append(mostfreqactivity)
                except:
                    likleyactivitiesdict[(startwindowtime, endwindowtime)] = [mostfreqactivity]

        # go through the results dict and make a new dict with only most likely activity across days

        finalactivitiesdict = {}

        for window in likleyactivitiesdict:
            activitieslist = likleyactivitiesdict[window]

            # get most freq activites, break with random on tie
            count = Counter(activitieslist)
            # get max count from result and then get all values that have that max count
            maxcount = max(count.values())
            mostcommonactlist = []
            for activity, count in count.items():
                if (count == maxcount):
                    mostcommonactlist.append(activity)
            randomactivity = mostcommonactlist[rand.randint(0, len(mostcommonactlist) - 1)]
            finalactivitiesdict[window] = randomactivity

        return finalactivitiesdict

    def getMostLikelyActivity(self, time):
        for timerange in self.mostlikelyactivities.keys():
            # check if time is within the time range
            if (time >= timerange[0] and time <= timerange[1]):  # time is in the range
                mostlikelyactivity = self.mostlikelyactivities[timerange]
                return mostlikelyactivity

    # this function generates timestamps for the outer activity sequence
    def generateActivityTimestamps(self, activitysequence, hiddenstatesequence, currenttime, reg):
        currentdayofweek = currenttime.date().weekday()
        curretimeofday = (currenttime - currenttime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        synthetictimestamps = []

        # start with first timestamp of real data to start the synthetic timestamp sequence
        synthetictimestamps.append(currenttime)
        # get timestamps for activity sequence
        previoushiddenstate = 0
        for i in range(len(activitysequence)):
            activity = activitysequence[i]
            hiddenstate = hiddenstatesequence[i]

            activitynum = self.activitynamemapstrtonum[activity]
            #features = preprocessing.normalize([[activitynum, currentdayofweek, curretimeofday, hiddenstate, previoushiddenstate]])
            features = [[activitynum, currentdayofweek, curretimeofday, hiddenstate, previoushiddenstate]]
            predduration = reg.predict(features)
            previoushiddenstate = hiddenstate

            #handle error in regression model
            #get max duration for this type of activity
            #get min duration for this type of activity
            minduration = min(self.activitydurationdist[activity])
            maxduration = max(self.activitydurationdist[activity])

            if(predduration[0] < minduration):
                predduration[0] = rand.choice(self.activitydurationdist[activity])
            if(predduration[0] > maxduration):
                predduration[0] = rand.choice(self.activitydurationdist[activity])

            seconds_in_day = 86400
            sum = curretimeofday + predduration[0]
            if(sum > seconds_in_day):
                curretimeofday = sum - 86400
            else:
                curretimeofday = sum

            currenttime = currenttime + datetime.timedelta(seconds=predduration[0])

            synthetictimestamps.append(currenttime)
            # convert seconds since midnight to timestamp

        return synthetictimestamps

    # this function generates the timesatmps for the sensor events
    def generateSensorEventTimestamp(self, activity,  currenttime, activitynum, sensornamenum, sensorstatenum, hiddenstate,
                                     previoushiddenstate, reg):

        try:
            currentdayofweek = currenttime.date().weekday()
        except AttributeError:
            print(currenttime)
            input("Issue here")
        curretimeofday = (currenttime - currenttime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        predduration = reg.predict([[activitynum, sensornamenum, sensorstatenum, currentdayofweek, curretimeofday,
                                     hiddenstate, previoushiddenstate]])

        mindur = min(self.sensoractivitydurationdict[activity])
        maxdur = max(self.sensoractivitydurationdict[activity])

        if(predduration[0] > maxdur):
            predduration[0] = rand.choice(self.sensoractivitydurationdict[activity])
        if(predduration[0] < mindur):
            predduration[0] = rand.choice(self.sensoractivitydurationdict[activity])

        currenttime = currenttime + datetime.timedelta(seconds=predduration[0])
        return currenttime

    # this function returns the duration between two timestamps in seconds
    def getDuration(self, row):
        currenttime = row["Date"]
        nexttime = row["NextDate"]

        duration = nexttime - currenttime
        durationseconds = duration / np.timedelta64(1, 's')

        return durationseconds

    # this function gets the HMM state that would be most likely to produce the real activity emission
    def getMostLikelyHiddenState(self, row):

        sensoreventhmm = sensoreventhmms[row["activity_name"]]
        # get the hmm model for the sensor event activity type
        sensor_event = row["sensor_name"] + "_" + row["sensor_state"]

        hidden_state = sensoreventhmm.getMostLikelyStateIndex(sensor_event)
        return hidden_state


    def run(self):

        global sensoreventhmms

        # First we need to generate the synthetic activity sequence
        self.syntheticactivitysequence, self.synthetichiddenstatesequence = self.generateActivitySequence(None, len(self.realactivitysequence))

        activitydatalist = []
        activityindexlist = []
        activitydfcolumns = ["activity_name", "activity_num", "day_of_week", "time_of_day", "duration", "hidden_state",
                             "previous_hidden_state"]

        for activity in self.activitynamesunique:
            self.activitydurationdist[activity] = []

        # We need the compressed activities and their start timestamps as an index in a dataframe
        # save the durations for each activity to a list for regression processing
        previoushiddenstate = 0
        for i in range(len(self.realactivitysequence) - 1):
            currenttimestamp = self.realactivitysequence.index[i]
            activityindexlist.append(currenttimestamp)

            datalist = []

            nexttimestamp = self.realactivitysequence.index[i + 1]
            activityinfodict = {}

            currentactivity = self.realactivitysequence[currenttimestamp]

            # activity_name
            datalist.append(currentactivity)

            # activity_num
            activitycode = self.realdataframe.loc[currenttimestamp]["activity_name_code"]
            datalist.append(activitycode)

            # dayofweek
            datalist.append(currenttimestamp.date().weekday())

            # timeofday
            datalist.append((currenttimestamp - currenttimestamp.replace(hour=0, minute=0, second=0,
                                                                         microsecond=0)).total_seconds())

            durationinseconds = nexttimestamp - currenttimestamp
            durationinseconds = durationinseconds / np.timedelta64(1, 's')

            self.activitydurationdist[self.realactivitysequence[i]].append(durationinseconds)


            # duration
            datalist.append(durationinseconds)

            # hidden state, get most likely state to emit symbol from the real sequence
            activity_name = self.realdataframe.loc[currenttimestamp]["activity_name"]
            hiddenstate = self.activityhmmmodel.getMostLikelyStateIndex(activity_name)
            datalist.append(hiddenstate)

            # previous hidden state
            datalist.append(previoushiddenstate)
            previoushiddenstate = hiddenstate

            activitydatalist.append(datalist)

        activitydurationdataframe = pd.DataFrame(activitydatalist, index=activityindexlist, columns=activitydfcolumns)

        # then we have all the info we need for the activity sequence

        # get the predicted durations for the original activity sequences
        # first I need to convert the information to the regression feature vectors
        # Feature vector for duration between activities is: <timeofday, dayofweek, activity, hiddenstate, previoushiddenstate> -> target <duration>
        # time of day is seconds since midnight
        # HMM states are most likely states to produce activities using the models trained on the real data

        # duration between activities features
        duractfeatures = activitydurationdataframe[
            ["activity_num", "day_of_week", "time_of_day", "hidden_state", "previous_hidden_state"]].values.tolist()
        duracttarget = activitydurationdataframe["duration"].values.tolist()

        # make a regression model for the activity durations
        reg = self.timestampmodel
        #duractfeatures = preprocessing.normalize(duractfeatures)
        duractfeatures = duractfeatures


        reg.fit(duractfeatures, duracttarget)

        # get synthetic timestamps
        currenttime = self.realactivitysequence.index[0]  # start timestamp, as time of day
        self.synthetictimestamps = self.generateActivityTimestamps(self.syntheticactivitysequence,
                                                                          self.synthetichiddenstatesequence, currenttime, reg)

        # now I have the first pass at synthetic timestamps and activity sequence
        # every time interval check most likely activity and regen sequence and timestamps

        # get most likely activities
        mostlikelyactivities = self.mostlikelyactivities

        # first attempt at synthetic activity sequence
        oldsequence = self.syntheticactivitysequence

        # then keep adding time interval and checking
        # regen the sequence with the most likely activity as the start state
        nextcheckpoint = self.realactivitysequence.index[0]
        for i in range(len(self.realactivitysequence)):
            timestamp = self.synthetictimestamps[i]
            activity = self.syntheticactivitysequence[i]

            # if we have reach a time interval mark
            # check the most likely activity for the current timestamp and regen the sequence
            if (timestamp >= nextcheckpoint):
                # get most likely acitivty for the time
                mostlikelyactivity = self.getMostLikelyActivity(timestamp)

                # resample
                numnewsamples = len(range(i, len(self.syntheticactivitysequence)))

                newsequence, newhiddenstatesequence = self.generateActivitySequence(mostlikelyactivity,
                                                                                                numnewsamples)

                # rewrite chunk of old sequence with new data
                for j in range(i, len(self.syntheticactivitysequence)):
                    # need i to start from 0 for the newsequence
                    self.syntheticactivitysequence[j] = newsequence[j - i]
                    self.synthetichiddenstatesequence[j] = newhiddenstatesequence[j - i]
                # get new timestamps for this regen sequence
                newtimestamps = self.generateActivityTimestamps(newsequence, newhiddenstatesequence,
                                                                            timestamp, reg)
                # merge with timestamp list
                for j in range(i, len(self.syntheticactivitysequence)):
                    self.synthetictimestamps[j] = newtimestamps[j - i]

                # get the next checkpoint
                nextcheckpoint = nextcheckpoint + datetime.timedelta(hours=self.timeinterval)

        # now we should have an improved activity sequence with matching timestamps

        # next we need to generate a synthetic seqeunce of sensor events for each activity
        syntheticsensoreventdataframe = self.generateSensorEventSequence(self.syntheticactivitysequence)

        # now get the durations for sensor events for the real data
        sensoreventdataframedict = {}

        # iterate through the data and fill the sensor event dict

        # now we have sensor events for each activity, we need to convert them to durations

        # now we have a dictionary for each activity and the sensor event durations

        # generate sensor event timestamps
        # first train the separate regression models
        # then get predictions

        self.realdataframe["sensor_duration"] = self.realdataframe.apply(self.getDuration, axis=1)

        sensoreventhmms = self.sensoreventhmms

        self.realdataframe["hidden_state"] = self.realdataframe.apply(self.getMostLikelyHiddenState,
                                                                                              axis=1)

        self.realdataframe["previous_hidden_state"] = self.realdataframe["hidden_state"].shift(
            1)

        # need to break into dict of activities and sensor event dataframes
        for activity in self.activitynamesunique:
            sensoreventdataframedict[activity] = \
            self.realdataframe.loc[self.realdataframe['activity_name'] == activity][
                ["sensor_duration", "time_of_day", "day_of_week", "activity_name", "activity_name_code", "sensor_name",
                 "sensor_name_code", "sensor_state", "sensor_state_code", "hidden_state",
                 "previous_hidden_state"]].dropna()

        sensoreventregressionmodeldict = {}
        self.sensoractivitydurationdict = {}
        for activity in self.activitynamesunique:
            sensoreventdurationframe = sensoreventdataframedict[activity]
            reg = linear_model.LinearRegression(normalize=True)
            seneventfeatures = sensoreventdurationframe[
                ["activity_name_code", "sensor_name_code", "sensor_state_code", "time_of_day", "day_of_week",
                 "hidden_state", "previous_hidden_state"]].values.tolist()
            seneventarget = sensoreventdurationframe["sensor_duration"].values.tolist()

            self.sensoractivitydurationdict[activity] = seneventarget


            reg.fit(seneventfeatures, seneventarget)
            sensoreventregressionmodeldict[activity] = reg
        # now we have regression models for the sensor events for each activity
        # go through the synthetic sequence and add timestamps for the sensor events
        # put the activity timestamps as the first timestamp for the block of sensor events

        activity_count = 0
        currentactivity = ""
        currenttime = ""
        nexttime = ""
        for i in range(1, len(syntheticsensoreventdataframe)):
            row = syntheticsensoreventdataframe.loc[i]
            # set to new activity
            if (row["activity_name"] != currentactivity):
                currentactivity = row["activity_name"]
                # get timestamp for the first sensor event in the activity

                currenttime = self.synthetictimestamps[activity_count]
                nexttime = self.synthetictimestamps[activity_count + 1]

                syntheticsensoreventdataframe.at[i, "timestamp"] = currenttime
                activity_count += 1
            else:
                # generate new time stamps for the rest of the sensor events
                # if the synthetic timestamp is greater than the timestamp for the next activity, add null value to dataframe to make sure they fit
                regmodel = sensoreventregressionmodeldict[row["activity_name"]]
                # generateSensorEventTimestamp(currenttime, activitynum, sensornamenum, sensorstatenum, hiddenstate, previoushiddenstate, reg)
                syntimestamp = self.generateSensorEventTimestamp(row["activity_name"], currenttime, int(row["activity_name_num"]),
                                                            int(row["sensor_name_num"]), int(row["sensor_state_num"]),
                                                            int(row["hidden_state"]), int(row["previous_hidden_state"]),
                                                            regmodel)

                if (syntimestamp < nexttime):
                    syntheticsensoreventdataframe.at[i, "timestamp"] = syntimestamp
                else:
                    syntheticsensoreventdataframe.at[i, "timestamp"] = ""
                currenttime = syntimestamp

        return syntheticsensoreventdataframe



