import os 
import sqlite3
import numpy as np
import ast 
import random 
import tensorflow as tf

parent_dir = os.path.dirname(os.path.abspath(__file__))

def select_data(num_samples = 1000):
    conn = sqlite3.connect('ecg_analysis')
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS selected_ecg_data;") 

    cursor.execute('''
        CREATE TABLE selected_ecg_data AS 
        SELECT image_file_path, time_series, arrhythmia_type FROM (
            SELECT image_file_path,  time_series,  arrhythmia_type, 
                   ROW_NUMBER() OVER (PARTITION BY arrhythmia_type ORDER BY RANDOM()) AS row_num
            FROM ecg_table WHERE image_file_path IS NOT NULL
        ) WHERE row_num <=  ?;
    ''', (num_samples, )) 

    conn.commit()
    conn.close()

def fetch_selected_data():
    conn = sqlite3.connect('ecg_analysis')
    cursor = conn.cursor()
    cursor.execute("SELECT image_file_path, time_series, arrhythmia_type FROM selected_ecg_data;")
    rows = cursor.fetchall()
    conn.close()
    img_paths, time_series, labels = zip(*rows)
    return list(img_paths), list(time_series), list(labels)

def format_images(img_paths):
    images = []
    for path in img_paths:
        img = tf.image.decode_png(tf.io.read_file(path), channels=1)  # grayscale
        img = tf.image.resize(img, (128, 128)) / 255.0  
        img = img.numpy()  
        img = np.reshape(img, (128, 128, 1))  
        images.append(img) 

    return np.array(images, dtype=np.float32)

def split_data(train_split=0.8, seed=42):
    """split the data deterministically"""
    
    # fetch the data
    img_paths , time_series, labels = fetch_selected_data()
    if not img_paths:
        print("No data found, skipping.")
        return None

    # shuffle the data
    data = list(zip(img_paths, time_series, labels) )
    random.seed(seed )
    random.shuffle( data)
    img_paths, time_series, labels = zip(*data)

    #convert to numpy arrays
    img_paths = np.array(img_paths) 
    time_series = [ast.literal_eval(ts) for ts in time_series]
    time_series  = np.array(time_series)  
    labels  = np.array(labels) 

    #split data
    split_ind = int(len(img_paths) * train_split)
    tr_imgs, val_imgs   = img_paths[:split_ind],  img_paths[split_ind:] 
    tr_series, val_series = time_series[:split_ind], time_series[split_ind:]
    tr_labels, val_labels = labels[:split_ind], labels[split_ind:]

    # integer encode l abels
    label_map = {label: idx for idx, label  in enumerate(sorted(set(labels)))}
    tr_labels =  np.array([label_map[label] for label in tr_labels], dtype=np.int32)
    val_labels = np.array([label_map[label]  for label in val_labels], dtype=np.int32)

    # convert file path to correctly formatted images
    tr_imgs = format_images(tr_imgs)
    val_imgs = format_images(val_imgs)

    return tr_imgs, val_imgs,  tr_series, val_series, tr_labels, val_labels, label_map
