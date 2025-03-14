import sqlite3 
import wfdb
import numpy as np 
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt 
from scipy.interpolate import lagrange 
import ast

import utils



labels = ['N', 'V', '/', 'R', 'L', 'A', '!', 'E']

def save_to_sqlite(database_path, record_paths, labels):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ecg_table (
        record_number INTEGER,
        arrhythmia_type TEXT,
        occurrence INTEGER,
        augmentation_no INTEGER, 
        timestamp INTEGER,        
        time_series TEXT,
        image_file_path TEXT,
        UNIQUE(record_number, arrhythmia_type, occurrence, augmentation_no)
    )
    ''')
 
    for record_path in record_paths:
        process_record(record_path, labels, cursor)
        conn.commit()

    conn.close()
    print("Data successfully saved to SQLite database.")

def process_record(record_path, labels, cursor):
    
    print(f"Processing record: {record_path}")
    
    #read the signal and annotation data
    signals, fields = wfdb.rdsamp(record_path, channels=[0])
    ann = wfdb.rdann(record_path, 'atr')  # Read annotations
    
    beats = ann.sample  # all beats (timestamps)
    types = ann.symbol  # all arrhythmia types
    record_number = record_number = os.path.basename(record_path)[:3]  # extract record name
    
    #dictionary to track occurrences of each arrhythmia type
    occurrence_counter = {label: 0 for label in labels}
    
    for idx, beat in enumerate(beats): 

        # determine the arrhythmia type for the current beat and skip if not required
        arrhythmia_type = types[idx]
        if arrhythmia_type not in labels: 
            continue
        
        #increment occurrence count for the type
        occurrence_counter[arrhythmia_type] = occurrence_counter.get(arrhythmia_type,  0) + 1
        
            
        # extract 128 samples to the left and right of the beat
        if beat > 128 and beat < len(signals) - 128:
            time_series = signals[beat - 128: beat +  128, 0]
        else:
            occurrence_counter[arrhythmia_type] -= 1
            continue  # skip if not enough data for slicing
        
        # insert data into the database
        cursor.execute('''
        INSERT OR IGNORE INTO ecg_table (
            record_number, 
            arrhythmia_type,
            occurrence, 
            augmentation_no,
            time_series
        ) VALUES (?, ?, ?, ?, ?)
    ''', (record_number, arrhythmia_type, occurrence_counter[arrhythmia_type], 0,
        str(time_series.tolist())))


def get_record_paths(): 
    cwd = os.getcwd()
    paths = glob.glob(os.path.join(cwd, "mit-bih-arrhythmia-database-1.0.0/*.atr"))
    paths = [path[:-4] for path in paths if '-' not in os.path.basename(path)]
    return paths

def poly_warp(time_series):
    """warp the series by a polynomial"""

    domain = len(time_series)

    #dummy values to be updated later
    poly_max, poly_min = 1, -1 
    poly = None  
    x_vals = None
    y_vals = None 
    x_interps = None
    y_interps = None

    #keep running until we generate a suitable polynomial
    while poly_max >= 0.15 or poly_min <= -0.15: 
        
        num_interp_points = 15    #number of points used to fit our lagrange polynomial

        #define the x positions for the points used to fit polynomial
        x_interps = []
        spacing = domain/(num_interp_points - 5)
        index = -2*spacing 
        for i in range(num_interp_points):
            x_interps.append(index)
            index += spacing 

        #fit the polynomial to an array of randomly generated points
        x_interps = np.array(x_interps)
        y_interps = np.random.uniform(-0.1, 0.1,  size=num_interp_points)
        poly = lagrange(x_interps,y_interps)

        # Generate a range of x values, excluding end points of lagrange
        # interpolation where the function becomes unstable
        x_vals = np.linspace(x_interps[1], x_interps[-2], 256)
        y_vals = poly(x_vals)

        #test that polynomial doesn't overmodify the data
        poly_max = max( y_vals) 
        poly_min = min(y_vals)

    warped_time_series = poly(x_vals)*(np.array(time_series)) + np.array(time_series)

    return warped_time_series


def x_axis_stretch(time_series):
    #crop a percentage off the ends of the time series and rescale

    crop_amount = np.random.uniform(0.01, 0.05)

    start_idx = int(crop_amount*len(time_series))
    end_idx = int((1-crop_amount)*len(time_series) )    
    cropped_series = time_series[start_idx:end_idx]

    original_indices = np.linspace(0, 1, len(cropped_series))
    new_indices = np.linspace(0, 1, len(time_series))

    rescaled_series = np.interp(new_indices, original_indices, cropped_series)

    return rescaled_series

def warp_and_stretch(time_series):
    #warp using lagrangre interpolation and crop and stretch along x-axis
    warped = poly_warp(time_series)

    warped_and_stretched = x_axis_stretch(warped) 

    return warped_and_stretched

def generate_augmented_samples(database_name, target_samples = 1000):
    #augment the data if there are too few samples

    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    
    for arrhythmia_type in labels :

        #find the number of samples for this arrhythmia type
        cursor.execute('SELECT COUNT(*) FROM ecg_table WHERE arrhythmia_type = ? ', (arrhythmia_type,))
        total_count = int(cursor.fetchone()[0])

        if total_count < target_samples: #generate if insufficient
            print(f"Insufficient samples, augmenting new samples for {arrhythmia_type}.")
            missing_count = target_samples - total_count 
            
            cursor.execute('''
            SELECT record_number, occurrence, time_series 
            FROM ecg_table 
            WHERE arrhythmia_type = ? AND augmentation_no = 0
            ''', (arrhythmia_type,) )
            rows = cursor.fetchall()

            # compute how many times each sample should be augmented
            aug_per_sample = ( (missing_count - 1) // len(rows) ) + 1 #base number of augmentations needed per sample
            
            new_records = []
            for record_number, occurrence, time_series in rows:
                time_series =  ast.literal_eval(time_series)  # Convert stored string to list

                # fetch the highest augmentation number for this record
                cursor.execute('''
                    SELECT MAX(augmentation_no) FROM ecg_table
                    WHERE record_number = ? AND occurrence = ? AND arrhythmia_type = ?
                ''', (record_number, occurrence, arrhythmia_type))
                current_max_aug = cursor.fetchone()[0]

                # generate the required augmentations per sample
                for i in range(aug_per_sample):
                    aug_series = warp_and_stretch(time_series)

                    # assign proper augmentation number
                    new_aug_num = current_max_aug + 1
                    current_max_aug += 1  # Increment for each new augmentation

                    new_records.append((
                        record_number, arrhythmia_type, occurrence, new_aug_num, str(aug_series.tolist())
                    ))

            
            cursor.executemany('''
                INSERT INTO ecg_table (
                    record_number,
                    arrhythmia_type, 
                    occurrence,
                    augmentation_no, 
                    time_series
                ) VALUES (?, ?, ?, ?, ?)
            ''', new_records)
            
            print(f"{arrhythmia_type}: {len(new_records)} augmented samples added.")
            conn.commit()
    
    conn.close()
    print("Augmentation process completed.")



def create_and_save_plots(database_name, output_base_dir):
    # ensure the base output directory exists
    os.makedirs(output_base_dir, exist_ok=True)    
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    
    # fetch all rows from ecg_table 
    cursor.execute("""SELECT record_number, arrhythmia_type, occurrence, augmentation_no, time_series 
        FROM ecg_table WHERE image_file_path IS NULL """) #don't bother running on already generated images
    rows = cursor.fetchall()
    
    for record_number, arrhythmia_type, occurrence, augmentation_no, time_series in rows:
        # replace "/" with "P" in folder path
        safe_label = "P" if arrhythmia_type == "/" else arrhythmia_type
        
        #create a subdirectory for the arrhythmia type if it doesn't exist
        arrhythmia_dir = os.path.join(output_base_dir, safe_label)
        os.makedirs(arrhythmia_dir, exist_ok=True)
        
        # convert time_series from string to list
        time_series = ast.literal_eval(time_series)
        
        # create a plot with no padding
        fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
        ax.plot(time_series, color='black', linewidth=0.75)
        ax.set_aspect('auto')
        ax.margins(0)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        #save the plot 
        file_name = f"{record_number}_{safe_label}_{occurrence}_{augmentation_no}.png"
        file_path = os.path.join(arrhythmia_dir, file_name)
        plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0 )
        plt.close()  # Close the figure to free memory
        
        #update the database with the plot file path
        cursor.execute('''
        UPDATE ecg_table
        SET image_file_path = ?
        WHERE record_number = ? AND arrhythmia_type = ? AND occurrence = ? AND augmentation_no = ?
        ''', (file_path, record_number, arrhythmia_type, occurrence, augmentation_no))
    
    conn.commit()
    conn.close()
    print(f"Plots saved to subdirectories in {output_base_dir} and database updated with file paths.")

def main():
    database_name = "ecg_analysis"
    save_to_sqlite(database_name, get_record_paths(), labels)
    generate_augmented_samples(database_name)
    create_and_save_plots(database_name, "plots")
    utils.select_data() #create a table containin only the desired data


if __name__ == "__main__":
    main()
