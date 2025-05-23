"""
DCNN (Deep Convolutional Neural Network) Model Execution
-------------------------------------------------------
Main script for running DCNN model including:
- Data preparation and splitting
- Feature processing
- Model evaluation
"""

import math, numpy as np
from DCNN import dcnn  # Main DCNN model implementation


def train_test_split(data, clas, tr_per):
    """
    Splits data into training and test sets while maintaining class balance
    
    Args:
        data: Input features
        clas: Class labels
        tr_per: Training percentage (0-100)
    
    Returns:
        train_x: Training features
        train_y: Training labels
        test_x: Test features
        test_y: Test labels
        label: All labels
    """
    train_x, train_y = [], []  # training data, training class
    test_x, test_y, label = [], [], []  # testing data, testing class, label
    uni = np.unique(clas)  # unique class

    for i in range(len(uni)):  # n_unique class
        tem = []
        for j in range(len(clas)):
            if (uni[i] == clas[j]):  # if class of data = unique class
                tem.append(data[j])  # get unique class as tem

        tp = int((len(tem) * tr_per) / 100)  # training data size

        for k in range(len(tem)):
            if (k < tp):  # adding training data & its class
                train_x.append(tem[k])
                train_y.append(float(uni[i]))
                label.append(float(uni[i]))
            else:  # adding testing data & its class
                test_x.append(tem[k])
                test_y.append(float(uni[i]))
                label.append(float(uni[i]))
    return train_x, train_y, test_x, test_y, label


def bound(f_data):
    """
    Reshapes feature data into square matrices for CNN input
    
    Args:
        f_data: Flattened feature vectors
        
    Returns:
        fe: Reshaped features as square matrices
    """
    fe = []
    sq = int(math.sqrt(len(f_data[0])))
    n = int(sq * sq)
    for i in range(len(f_data)):
        tem = []
        for j in range(n):  # attributes in each row
            tem.append(f_data[i][j])  # add value to tem array
        fe.append(tem)  # add 1 row of array value to fe
    return fe




def callmain(data, label, trp, acc, sen, spe):
    """
    Main DCNN evaluation function
    
    Args:
        data: Input features
        label: Ground truth labels
        trp: Training percentage
        acc: List to store accuracy results
        sen: List to store sensitivity results
        spe: List to store specificity results
    
    Calculates and stores:
    - Accuracy (acc)
    - Sensitivity (sen)
    - Specificity (spe)
    """
    train_x, train_y, test_x, test_y, target = train_test_split(data, label, trp)  # splitting training & testing data
    feature = np.asarray(bound(data))
    feature = feature.astype('float')
    y_pred = dcnn.classify(np.array(feature), np.array(target), np.array(train_y), np.array(test_y), trp)
    target = test_y
    unique_clas = np.unique(test_y)
    tp, tn, fn, fp = 0, 0, 0, 0
    pred_val = np.unique(target)
    for i1 in range(len(unique_clas)):
        # c = unique_clas[i1]
        c = unique_clas[i1]
        for i in range(len(target)):
            if (target[i] == c and y_pred[i] == c):
                tp = tp + 1
            if (target[i] != c and y_pred[i] != c):
                tn = tn + 1
            if (target[i] == c and y_pred[i] != c):
                fn = fn + 1
            if (target[i] != c and y_pred[i] == c):
                fp = fp + 1
    tn = tn / len(pred_val)
    tp = tp / len(pred_val)
    fn = fn / pred_val[len(pred_val) - 1]
    fp = fp / pred_val[len(pred_val) - 1]
    tn = tn / len(unique_clas)
    sen.append(tp / (tp + fn))
    spe.append(tn / (tn + fp))
    acc.append((tp + tn) / (tp + tn + fp + fn))
    acc.sort()
    spe.sort()
    sen.sort()
