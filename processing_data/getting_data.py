import os
import glob

def get_records():
    """ setting the path for the data """
    print("hello world!")

    script_dir = os.path.dirname(__file__)
    file_regex = script_dir+'/mit-bih-arrhythmia-database-1.0.0/*.atr'

    print(file_regex)
    print(type(file_regex))

    paths = glob.glob(file_regex)

    print(paths)

get_records()