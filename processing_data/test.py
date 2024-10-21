import matplotlib.pyplot as plt
import wfdb
import numpy as np
import os
from pathlib import Path
import glob
from pprint import pprint

#global variable containing arryhtmia labeling system in dataset
arrhythmia_labels = ['N', 'V', '/', 'R', 'L', 'A', '!', 'E'] 

def get_record_paths(): 
    cwd = os.getcwd()
    paths = glob.glob(os.path.join(cwd, "mit-bih-arrhythmia-database-1.0.0/*.atr"))
    paths = [path[:-4] for path in paths if '-' not in os.path.basename(path)]
    
    return paths

def _segmentation(record,label):
    """
    Takes a single record which is broken up into .atr, .head, .dat and .xws 
    files, filters looking for a particular arrythmia type in the .atr file, 
    cross references it with the .head file and returns an array of the time series
    analysis from the file which fits the labeling. For example, when fed a reference
    for record 230 and the label normal, it will return an array of  time series 
    analyses for each of the normal beats in the file.  
    """

    type_array = []
    for e in record:
        signals, fields = wfdb.rdsamp(e, channels = [0]) 

        ann = wfdb.rdann(e, 'atr')
        arr_type = [label]
        ids = np.in1d(ann.symbol, arr_type)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                type_array.append(signals[beats[j] - diff1: beats[j] + diff2, 0])    
    return type_array

def segment_record(record):
    """takes a record a and splits it up into the different kinds of 
    fibrilations."""

    segmented_record = [] #entry corresponds to the time series of each type

    for label in arrhythmia_labels:
        segmented_record.append(_segmentation([record], label))

    return segmented_record


def main():
    paths = get_record_paths()
    create_folders()

    records_dict = {} #store the rec_no and segmented_rec number pairs
    for path in paths: 
        rec_no = os.path.basename(path)[:3] #record number

        segmented_record = segment_record(path) #segmented record associated to each record

        records_dict[rec_no] = segmented_record

    #now plot and save each time series to the appropiate place
    for rec_no, segmented_record in records_dict.items():
        index_arr_type = 0 #reference to the fibrilation type in arrythmia labels
        for arr_type in segmented_record:

            type_label = arrhythmia_labels[index_arr_type] #get the string value of the label
            count = 1 #number of times a type of fibrilation appears in the record
            
            for time_series in arr_type:
                save_plot(time_series, count, rec_no, type_label)
                count = count + 1

            index_arr_type += 1

def save_plot(time_series, count, rec_no, arr_type):
    print(time_series)
    plt.plot(time_series)
      
    
def create_folders():
    print("here")
    cwd = Path.cwd() 
    image_folder = cwd / "image_data" 
    image_folder.mkdir(exist_ok=True)

    for subfolder in arrhythmia_labels:
        subfolder_path = image_folder / subfolder
        subfolder_path.mkdir(exist_ok=True)

main()