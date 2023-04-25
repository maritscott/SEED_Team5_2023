#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:10:24 2023

The purpose of this code is to establish the foundations of an algorithm that identifies kicks using data from
a triaxial accelerometer. 

The code first imports training datasets, then finds a threshold to define kicks from gentle motions.
Next, the threshold is tested using imported testing datasets. sensitivity and specificity are calculated.

Finally, the test data is plotted gradually to simulate what a real-time algorithm would look like.

Before running:
Save TrainData and TestData folders to working directory


@author: maritscott
UVM SEED 2023- Team 5
"""


#%% Import Packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import os
import glob

#%% Define Variables

fs = 10  # samples per second 
l1_duration= 60
l23_duration = 40
epoch_duration = 5 #seconds
epoch_length = epoch_duration* fs # samples

#%% Define functions

def epoch_data(data, num_segments, segment_length_samp, num_cols):
    """
    This function generates a 3D array from a 2D array.

    Parameters
    ----------
    data : 2D numpy array
        contains one trial of accelerometer data. rows = time, columns = x, y, z and mag.
    num_segments : int
        desired number of pages in the epoch.
    segment_length_samp : int
        desired number of rows in each page.
    num_cols : int
        number of columns in dataset.

    Returns
    -------
    data_epoch : 3D numpy array
        data converted into a 3d array.

    """
    data_epoch = np.zeros([num_segments, segment_length_samp, num_cols]) 
        
    for segment_index in range(num_segments): 
        start = segment_index*segment_length_samp
        end = start + segment_length_samp
        data_epoch[segment_index, :,:] = data.iloc[start:end,: ]
        
    return data_epoch


def get_segment_maxes(data_epoch, num_segments, segment_length_samp, column_of_int):
    """
    This function calculates the maximum magnitudes of a data epoch and their indices.

    Parameters
    ----------
    data_epoch : 3D numpy array
        DESCRIPTION.
    num_segments : int
        number of pages in epoch.
    segment_length_samp : int
        number of rows in each page.
    column_of_int : int
        column you wish to perfom calculation on.

    Returns
    -------
    max_vals : 2d numpy array
        array of maximum values from each page of the epoch.
    max_ids : 2s numpy array
        array of indices from maximum values from each page of the epoch.

    """
    max_vals = []
    max_ids = []
    
    for segment_index in range(num_segments):
        data_segment = data_epoch[segment_index, :, column_of_int]
        max_vals = max_vals + [np.max(data_segment)] # max of each segment 
        max_ids = max_ids + [np.argmax(data_segment)+(segment_index*segment_length_samp)] # location of max of each segment 
        
    return max_vals, max_ids

#%% Import level 1 data

all_train_data = dict()
     
# read in sample data from csv
data_path = r'TrainData'     
# level 1 data is shorter than levels 2 and 3, so they will be imported separately                                                    
level1_files = sorted(glob.glob(os.path.join(data_path, "*.csv"))) 
level23_files = sorted(glob.glob(os.path.join(data_path, "*.CSV"))) 
csv_files = level1_files + level23_files
csv_files = sorted(csv_files)

col_names = ['time', 'x', 'y', 'z']

num_datasets = len(csv_files)
dataset_names = []
for file_index in range(num_datasets): 
    dataset_name = csv_files[file_index][10:-4]
    dataset_names = dataset_names + [dataset_name]
    # load data from current csv
    csv_data = pd.read_csv(csv_files[file_index], header=None, names = col_names) 
    # convert to g
    csv_data.iloc[:,1:4] = (.244/1000)*csv_data.iloc[:,1:4]
    csv_data.rename(columns = {'1':'time', '2':'x', '3':'y', '4':'z'}, inplace = True)
    
    
    curr_data = csv_data
    
    x = curr_data['x']
    y = curr_data['y']
    z = curr_data['z']
    
    # calculate magnitude and store
    
    x2 = x**2
    y2 = y**2
    z2 = z**2
    
    s = x2+y2+z2
    
    mag = np.sqrt(s)
    
    
    curr_data['mag'] = mag
    
    if ('1' in dataset_name):
        #print(dataset_name)
        all_train_data[file_index] = {'name': dataset_name, 
                                        'data': curr_data.iloc[0:l1_duration*fs,:]}
    else:
        #print(dataset_name)
        all_train_data[file_index] = {'name': dataset_name, 
                                        'data': curr_data.iloc[0:l23_duration*fs,:]}



#%% Plot all datasets in their own figure

for dataset_index in all_train_data:
    data = all_train_data[dataset_index]['data']
    plt.figure()
    plt.plot(data['time'],data.iloc[:,1:5])
    plt.legend(data.iloc[:,1:])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (g)')
    plt.title(all_train_data[dataset_index]['name'])
    
#%% Plot all datasets together (just magnitude)

plt.figure(100)
legend = []
for dataset_index in all_train_data:
    data = all_train_data[dataset_index]['data']
    name = all_train_data[dataset_index]['name'] 
    legend += [name]
    plt.plot(data['time'],data['mag'])    
    #plt.plot(all_data['time'],mag)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (g)')
    #plt.title(all_train_data[dataset_index]['name'])
    print(len(data))
plt.legend(legend)    
plt.grid()
plt.show()



#%% Get max magnitudes for each segment and store in dict
    
total_time_1 = 60 # seconds
segment_duration_1 = 10 # seconds
total_samples_1 = fs*total_time_1
num_segments_1 = int(total_time_1/segment_duration_1)
segment_length_samp_1 = int(segment_duration_1 * fs)
num_cols =  5 # - x,y,z, time, mag

total_time_2 = 40 # seconds
segment_duration_2 = 5 # seconds
total_samples_2 = fs*total_time_2
num_segments_2 = int(total_time_2/segment_duration_2)
segment_length_samp_2 = int(segment_duration_2 * fs)

column_of_int = 4 # magnitude column is of interest



# loop through trials in all_train_data
for dataset_index in range(num_datasets):
    data = all_train_data[dataset_index]['data']
    name = all_train_data[dataset_index]['name'] 
    
    # level 1 datasets 
    if ('1' in name):
        # epoch trial 
        data_epoch = epoch_data(data, num_segments_1, segment_length_samp_1, num_cols)
        all_train_data[dataset_index]['data_epoch'] = data_epoch
    
        # loop through segments in trial
        for segment_index in range(num_segments_1):
            # get arr of max in each segment
            max_vals, max_ids = get_segment_maxes(data_epoch, num_segments_1, segment_length_samp_1, column_of_int)
            all_train_data[dataset_index]['max_vals'] = max_vals
            all_train_data[dataset_index]['max_ids'] = max_ids

    
    # level 2,3 datasets 
    else:
        # epoch trial 
        data_epoch = epoch_data(data, num_segments_2, segment_length_samp_2, num_cols)
        all_train_data[dataset_index]['data_epoch'] = data_epoch
    
        # loop through segments in trial
        for segment_index in range(num_segments_2):
            # get arr of max in each segment
            max_vals, max_ids = get_segment_maxes(data_epoch, num_segments_2, segment_length_samp_2, column_of_int)
            all_train_data[dataset_index]['max_vals'] = max_vals
            all_train_data[dataset_index]['max_ids'] = max_ids

            
    # plot to confirm maxes worked
    plt.figure()
    plt.plot(data['mag'])
    plt.plot(max_ids, max_vals, 'r*')
            
  
#%% Create data frame of all max values for all segments all datasets

max_results = pd.DataFrame(np.empty([num_segments_2, num_datasets]), 
                           columns=dataset_names)
max_results[:] = np.nan

# fill in values
for dataset_index in range(num_datasets):
    max_vals = all_train_data[dataset_index]['max_vals']
    name = all_train_data[dataset_index]['name'] 
    
    if ('1' in name):
        max_results.iloc[0:num_segments_1,dataset_index] = max_vals
        max_results.iloc[6:,dataset_index] = np.mean(max_vals)
        
    
    else:
        max_results.iloc[:,dataset_index] = max_vals 

max_results_avg = pd.DataFrame(np.empty([num_segments_2, 3]), 
                           columns=['level 1', 'level 2', 'level 3'])

max_results_avg['level 1'] = (max_results['IO_1'] + max_results['SS_1'] + max_results['UD_1'])/3
max_results_avg['level 2'] = (max_results['IO_2'] + max_results['SS_2'] + max_results['UD_2'])/3
max_results_avg['level 3'] = (max_results['IO_3'] + max_results['SS_3'] + max_results['UD_3'])/3


#%% Box plots for all kick types 
plt.figure()
plt.boxplot(max_results, labels=list(max_results.columns))
plt.title('Max magnitude for each type of kick')
plt.ylabel('Acceleration (g)')
plt.xlabel('Kick Type')



#%% Establish threshold

level1_max_max = np.max(max_results_avg['level 1'])
level2_max_min = np.min(max_results_avg['level 2'])

max_threshold = ((level1_max_max*7) + level2_max_min)/8 # take weighted average of the boundaries of level 1 and level 2, favoring level 1
                
plt.figure()
plt.boxplot(max_results_avg, labels=list(max_results_avg.columns))
plt.title('Max Magnitude of 3 Kick Levels')
plt.ylabel('Acceleration (g)')
plt.hlines(max_threshold, 0,4)



#%% Input test data

all_test_data = dict()

#read in sample data from csv
data_path = r'TestData'                                                          # define relative path where data is located
csv_files = sorted(glob.glob(os.path.join(data_path, "*.CSV"))) 

col_names = ['time', 'x', 'y', 'z']

num_datasets = len(csv_files)
dataset_names = []
for file_index in range(num_datasets): 
    dataset_name = csv_files[file_index][9:-4]
    dataset_names = dataset_names + [dataset_name]
    # load data from current csv
    csv_data = pd.read_csv(csv_files[file_index], header=None, names = col_names) 
    # convert to g
    csv_data.iloc[:,1:4] = (.244/1000)*csv_data.iloc[:,1:4]

    csv_data.rename(columns = {'1':'time', '2':'x', '3':'y', '4':'z'}, inplace = True)
    
    
    curr_data = csv_data
    
    x = curr_data['x']
    y = curr_data['y']
    z = curr_data['z']
    
    
    x2 = x**2
    y2 = y**2
    z2 = z**2
    
    s = x2+y2+z2
    
    mag = np.sqrt(s)
    
    
    curr_data['mag'] = mag
    
    
    all_test_data[file_index] = {'name': dataset_name, 
                                        'data': curr_data}

#%% self input truth arrays based on known truths of testing data trials

all_test_data[0]['is_kick_truth'] = np.array([False, True, True, False, True, True, False, True, True])
all_test_data[1]['is_kick_truth'] = np.array([True, True, False, True, True, False, True, True, False])
all_test_data[2]['is_kick_truth'] = np.array([True, False, True, True, False, True, True, False, True,])

#%% plot to confirm test data was imported correctly

plt.figure()
legend = []
for dataset_index in all_test_data:
    data = all_test_data[dataset_index]['data']
    name = all_test_data[dataset_index]['name'] 
    legend += [name]
    plt.plot(data['time'],data['mag'])    
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (g)')
    plt.title(all_test_data[dataset_index]['name'])
    print(len(data))
plt.legend(legend)    
plt.grid()
plt.show()



#%% Epoch, get max, and create confusion matrices for test data

total_time = 45 # seconds
segment_duration = 5 # seconds
total_samples = fs*total_time
num_segments = int(total_time/segment_duration)
segment_length_samp = int(segment_duration * fs)

column_of_int = 4

for dataset_index in range(num_datasets):
    data = all_test_data[dataset_index]['data']
    name = all_test_data[dataset_index]['name'] 
    
    
    # epoch trial 
    data_epoch = epoch_data(data, num_segments, segment_length_samp, num_cols)
    all_test_data[dataset_index]['data_epoch'] = data_epoch

    # loop through segments in trial
    for segment_index in range(num_segments_1):
        # get arr of max in each segment
        max_vals, max_ids = get_segment_maxes(data_epoch, num_segments, segment_length_samp, column_of_int)
        all_test_data[dataset_index]['max_vals'] = max_vals
        all_test_data[dataset_index]['max_ids'] = max_ids
        
        max_vals = np.array(max_vals)
        max_ids = np.array(max_ids)
        
     
    is_kick_pred = np.array(max_vals>max_threshold)
    all_test_data[dataset_index]['is_kick_pred'] = is_kick_pred
    is_kick_truth = all_test_data[dataset_index]['is_kick_truth']

    max_vals_k = max_vals[is_kick_truth]
    max_ids_k = max_ids[is_kick_truth]
    
    # plot to confirm maxes worked
    plt.figure()
    plt.plot(data['mag'])
    plt.hlines(max_threshold, 0, total_samples, 'y')
    plt.plot(max_ids, max_vals, 'r*' )
    plt.plot(max_ids_k, max_vals_k, 'gs', markerfacecolor='none')
    plt.title(name)
    plt.xlabel('# samples')
    plt.ylabel('Acceleration (g)')
    plt.legend(['Acceleration magnitude', 'Threshold', 'Max magnitude in 5 sec segment', 'Identified kick'])
    
    
    # confusion matrix
    actions = [True, False]
  
    conf_mat_max=np.empty((len(actions),len(actions)) , dtype = int)
    
    for row_index in range(len(actions)):
        for col_index in range(len(actions)):
            conf_mat_max[row_index,col_index]= np.sum(np.logical_and((is_kick_pred == actions[row_index]), (is_kick_truth == actions[col_index])))
              
    
    plt.figure()
    plt.imshow(conf_mat_max)
    plt.ylabel('predicted ')
    plt.xlabel('actual ')
    plt.xticks(np.arange(2), ['Kick', 'Not Kick'])
    plt.yticks(np.arange(2), ['Kick', 'Not Kick'])
    cbar = plt.colorbar()
    cbar.set_label('# motions')
    plt.title(f'{name} Confusion Matrix')
    
    for (j,i),label in np.ndenumerate(conf_mat_max):
        plt.text(i,j,label,ha='center',va='center', color = 'white')
        plt.text(i,j,label,ha='center',va='center', color = 'white')
    
    
    TP = conf_mat_max[0,0]
    FN = conf_mat_max[1,0]
    FP = conf_mat_max[0,1]
    TN = conf_mat_max[1,1]
    
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    
    print(f'The sensitivity of {name} is {sensitivity}')
    print(f'The specificity of {name} is {specificity}')
    print('-----')
    
    



#%% Combine the three conf mats into 1

all_truth = np.append(all_test_data[0]['is_kick_truth'], [all_test_data[1]['is_kick_truth'],all_test_data[2]['is_kick_truth']])
all_pred = np.append(all_test_data[0]['is_kick_pred'], [all_test_data[1]['is_kick_pred'],all_test_data[2]['is_kick_pred']])

# confusion matrix
actions = [True, False]

conf_mat_max=np.empty((len(actions),len(actions)) , dtype = int)

for row_index in range(len(actions)):
    for col_index in range(len(actions)):
        conf_mat_max[row_index,col_index]= np.sum(np.logical_and((all_pred == actions[row_index]), (all_truth == actions[col_index])))
          

plt.figure()
plt.imshow(conf_mat_max)
plt.ylabel('predicted ')
plt.xlabel('actual ')
plt.xticks(np.arange(2), ['Kick', 'Not Kick'])
plt.yticks(np.arange(2), ['Kick', 'Not Kick'])
cbar = plt.colorbar()
cbar.set_label('# motions')
plt.title('All Test Trials Confusion Matrix')

for (j,i),label in np.ndenumerate(conf_mat_max):
    plt.text(i,j,label,ha='center',va='center', color = 'red')


TP = conf_mat_max[0,0]
FN = conf_mat_max[1,0]
FP = conf_mat_max[0,1]
TN = conf_mat_max[1,1]

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

print(f'The sensitivity of all testing data is {sensitivity}')
print(f'The specificity of all testing data is {specificity}')
print('-----')






#%% Update plots gradually to simulate a realtime algorithm 

# choose a dataset to plot 
data = all_test_data[2]['data']

# create empty lists which will be filled in gradually

time_data = []
xdata = []
ydata = []
zdata = []
magdata = []
kicks_time = []
kicks = []


y_max = 1.2 *max((data['mag']))

fig, (axes1) = plt.subplots(1,1,num=3,figsize=[8,12])

linex, = axes1.plot(time_data, xdata, 'r-')
liney, = axes1.plot(time_data, ydata, 'b-')
linez, = axes1.plot(time_data, zdata, 'g-')
linem, = axes1.plot(time_data, magdata, 'black')
linek, = axes1.plot(kicks_time, kicks, '*')

axes1.set_xlim(0, total_time)
axes1.set_ylim(0, y_max )
axes1.set_xlabel('Time (s)')
axes1.set_ylabel('Acceleration (g)')
axes1.set_title('Accelerometer Data', size = '10')
axes1.hlines(max_threshold, 0, len(data), 'y')


# loop through samples in test dataset 

segment_count = -1
for sample_index in range(total_samples):
    time_data.append(data.iloc[sample_index,0])
    magdata.append(data.iloc[sample_index,4]) #magnitude
    
   
    linem.set_xdata(time_data)
    linem.set_ydata(magdata)
    

     # check every epoch_duration seconds to see if the magnitude is above the threshold   
    if (sample_index % epoch_length == 0 and sample_index >= epoch_length):
        segment_count = segment_count+1
        max_in_epoch = np.max(data['mag'][sample_index-epoch_length:sample_index])
        max_index = np.argmax(data['mag'][sample_index-epoch_length:sample_index])+(segment_count*epoch_length)
        max_time = max_index/fs
        if (max_in_epoch > max_threshold):
            kicks_time.append(max_time)
            kicks.append(max_in_epoch)
            
            linek.set_xdata(kicks_time)
            linek.set_ydata(kicks)

    plt.show()
    plt.pause(.1)
    







