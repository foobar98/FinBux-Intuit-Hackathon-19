import warnings
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from xgboost import plot_importance
import pickle

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load arff dataset
def load_bankruptcy_arff():
    num_files = 5 
    load_bankruptcy_arff = [arff.loadarff('data/' + str(i + 1) + 'year.arff') for i in range(num_files)]
    return load_bankruptcy_arff

def arff_to_dataframe():
    return [pd.DataFrame(data_i_year[0]) for data_i_year in load_bankruptcy_arff()]

def set_column_names(df_bankruptcy):
    cols = ['X' + str(i + 1) for i in range(len(df_bankruptcy[0].columns) - 1)]
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

def model_train():
    global df_bankruptcy
    global f

    f = open("first_run.txt","w+")

    df_bankruptcy = arff_to_dataframe()
    set_column_names(df_bankruptcy)
    convert_datatype(df_bankruptcy)
    labels_to_binary(df_bankruptcy)

    features_df, labels_df = split_features_labels(df_bankruptcy)

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(features_df[0], labels_df[0], test_size=test_size, random_state=seed)

    # train model    
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # save model to file
    pickle.dump(model, open("model.pickle.dat", "wb"))

    y_pred = model.predict(X_test)
    predictions = [np.round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1score = (2*precision*recall)/(precision+recall)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%%" % (precision * 100.0))
    print("Recall: %.2f%%" % (recall * 100.0))
    print("F1 Score: %.2f%%" % (f1score * 100.0))

    # make current company prediction
    X_test = features_df[0].tail(1)
    x_df = pd.DataFrame(X_test)
    column_list = [] 
    for i in range(1,65):
        column_list.append("X"+str(i))
    x_df.columns = column_list
    X_test = x_df  
    
    my_pred = model.predict_proba(X_test)

    # save company prediction in file
    f.write("var first = " + str(my_pred[0][0]))
    print(my_pred)
    f.close()
