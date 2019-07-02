def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from scipy.io import arff
import random
from collections import OrderedDict
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from xgboost import XGBClassifier
from xgboost import plot_importance
import pickle

def load_bankruptcy_arff():
    num_files = 5  # 5 files for company information with class labels
    load_bankruptcy_arff = [arff.loadarff('data/' + str(i + 1) + 'year.arff') for i in range(num_files)]
    return load_bankruptcy_arff


# Load arff files into pandas

def arff_to_dataframe():
    return [pd.DataFrame(data_i_year[0]) for data_i_year in load_bankruptcy_arff()]


def set_column_names(df_bankruptcy):
    cols = ['X' + str(i + 1) for i in range(len(df_bankruptcy[0].columns) - 1)]  # x1 to X64
    cols.append('Y')
    for df in df_bankruptcy:
        df.columns = cols

def convert_datatype(df):
    for i in range(5):
        index = 1
        while (index <= 63):
            colname = df[i].columns[index]
            col = getattr(df[i], colname)
            df[i][colname] = col.astype(float)
            index += 1

# Converting label column to 0 or 1
def labels_to_binary(df):
    for i in range(len(df)):
        col = getattr(df[i], 'Y')
        df[i]['Y'] = col.astype(int)

# Splitting features and labels
def split_features_labels(dfs):
    feature_dfs = [dfs[i].iloc[:, 0:64] for i in range(len(dfs))]
    label_dfs = [dfs[i].iloc[:, 64] for i in range(len(dfs))]
    return feature_dfs, label_dfs


def solve():
    global df_bankruptcy

    df_bankruptcy = arff_to_dataframe()
    print()
    set_column_names(df_bankruptcy)
    convert_datatype(df_bankruptcy)
    labels_to_binary(df_bankruptcy)

    features_df, labels_df = split_features_labels(df_bankruptcy)
    

solve()