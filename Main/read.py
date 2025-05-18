"""
Data Reading Utilities
---------------------
Provides functions for reading:
- Feature vectors from Feat.csv
- Labels from Label.csv
"""

import csv
import numpy as np


def read_data():
    """
    Reads feature vectors from Feat.csv
    
    Returns:
        List of lists containing feature vectors
        Each inner list represents one sample's features
    """
    import os
    file_path = os.path.join(os.path.dirname(__file__), "Feat.csv")  # Feature data file
    print(f"Reading features from: {file_path}")  # Debug: show file path
    print(f"Current working directory: {os.getcwd()}")  # Debug: show working dir
    print(f"File exists: {os.path.exists(file_path)}")  # Debug: check file existence
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feat.csv not found at: {file_path}")

    datas = []
    with open(file_path, 'rt') as f:
        content = csv.reader(f)                        #read csv content
        for rows in content:                           #row of data
            tem = []
            for cols in rows:                          #attributes in each row
                tem.append(float(cols))             #add value to temporary array
            datas.append(tem)                         #add 1 row of array value to dataset
    return datas

def read_label():
    """
    Reads class labels from Label.csv
    
    Returns:
        List of integer labels (0 or 1)
        Each element represents one sample's class
    """
    import os
    file_path = os.path.join(os.path.dirname(__file__), "Label.csv")  # Label data file
    print(f"Reading labels from: {file_path}")  # Debug: show file path
    print(f"Current working directory: {os.getcwd()}")  # Debug: show working dir
    print(f"File exists: {os.path.exists(file_path)}")  # Debug: check file existence
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label.csv not found at: {file_path}")

    datas = []
    with open(file_path, 'rt') as f:
        content = csv.reader(f)                        #read csv content
        for rows in content:                           #row of data
            for cols in rows:                          #attributes in each row
                datas.append(int(float(cols)))  # add value to temporary array
    return datas
