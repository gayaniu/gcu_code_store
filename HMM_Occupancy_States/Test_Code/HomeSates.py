import pandas as pd
import json


print("Space Data")
space_data = pd.read_json (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_spaces.json')
print(space_data)


#parsed_json = (json.loads(r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_spaces.json'))
#print(json.dumps(parsed_json, indent=4, sort_keys=True))

with open(r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_spaces.json') as f:
    distros_dict = json.load(f)

for distro in distros_dict:
    print(distro)


df = pd.DataFrame.from_dict(distros_dict, orient='columns')
print(df)

#########################################################
##read the devices data file#############################

device_data = pd.read_json (r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_devices.json')
print("Device Data")
print(device_data)

###Fltering the device data from the devices, device ID = 2 ,18 ,17 , here we consider the entire house as space and then get the devices in exit and entry points - considering the front and the back doors. 

device_list = [2,18,17]
device_data_entire_space =  device_data[device_data['id'].isin(device_list)]

print (device_data_entire_space)


####################################################################################
###Read the 2017/01/27  dataset and get the above mentioned device data##############

Sensor_data_01_27 = pd.read_csv(r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_2017-01-27_events.csv')
Sensor_data_01_27 = Sensor_data_01_27 [Sensor_data_01_27['sensor.sensor_id'].isin(device_list)]
print(Sensor_data_01_27)


######Read the full dataset ####################################################################
Sensor_data = pd.read_csv(r'D:\Masters in Data Science\2nd Year\Trimester 02\SIT790\HMM Home States\Data\P1_full_events.csv')
#Sensor_data = Sensor_data [Sensor_data['id'].isin(device_list)]
#print(Sensor_data)

print("printig to check missing values")
print(Sensor_data.isnull())


#check the datatypes
print("printig tdata types")
print(Sensor_data.dtypes)

#takeing the data types object in to consideration
obj_df = Sensor_data.select_dtypes(include=['object']).copy()
obj_df.head()


#Checking the null values 
print("printig null values")
print(obj_df[obj_df.isnull().any(axis=1)])

#Checking the null values 
print("Checking the state values")
obj_df["state"].value_counts()


#Changing the State variable in to int 64 and repalce active = 1 and inactive  =0 
cleanup_nums = {"state": {"active": 1, "inactive": 0, "open": 2, "closed" : 3}}
obj_df.replace(cleanup_nums, inplace=True)
print(obj_df)

cleanup_nums_Type = {"type": {"contact": 1, "motion": 0}}
obj_df.replace(cleanup_nums_Type, inplace=True)
print(obj_df)


print(obj_df.dtypes)
#print (obj_df.values.reshape(-1,1))

#Applying k-means to find the best 


from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#Data
boston = datasets.load_boston()
ts_data = boston.data[1,:]
#HMM Model
gm = hmm.GaussianHMM(n_components=3)
gm.fit(ts_data.reshape(-1, 1))
states = gm.predict(ts_data.reshape(-1, 1))
#Plot
color_dict = { 0:"r",1:"g",2:"b" }
color_array = [ color_dict[i] for i in states ]
plt.scatter(range(len(ts_data)), ts_data, c=color_array)
plt.title("HMM Model")













