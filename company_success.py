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

f = open("model_output.txt","w+")

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

############################################################
# dataframes is the list of pandas dataframes for the 5 year datafiles.


# Change column names for easier coding : x1 to x64 attributes/columns and Y as label column that states company is bankrupt or not



# print(df_bankruptcy[0].head())

# print(type(df_bankruptcy[0]))

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





# Preprocessing data: Handling missing values and/or NA's

def drop_nans(df_bankruptcy, verbose=False):
    clean_dataframes = [df.dropna(axis=0, how='any') for df in df_bankruptcy]
    return clean_dataframes


def mean_imputation(df):
    # Construct an imputer with strategy as 'mean', to mean-impute along the columns
    imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    mean_imputed_dfs = [pd.DataFrame(imputer.fit_transform(df)) for df in df_bankruptcy]
    for i in range(len(df_bankruptcy)):
        mean_imputed_dfs[i].columns = df_bankruptcy[i].columns
    return mean_imputed_dfs


# Best practice to handle missing data is to substitute it with different values
# In this case, performing imputation with mean values


# print(mean_imputed_dataframes[1].head(5))

# Splitting features and labels
def split_features_labels(dfs):
    feature_dfs = [dfs[i].iloc[:, 0:64] for i in range(len(dfs))]
    label_dfs = [dfs[i].iloc[:, 64] for i in range(len(dfs))]
    return feature_dfs, label_dfs


# Classifiers for the dataset

# K-Fold Cross Validation
def kfold_cv(k, X, y, verbose=False):
    X = X.values  # Features
    y = y.values  # Labels
    kf = KFold(n_splits=k, shuffle=False, random_state=42)
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    return X_train, y_train, X_test, y_test





def perform_data_modeling(_models_, _imputers_, verbose=False, k_folds=5):
    # 7 metrics, averaged over all the K-Folds
    model_results = OrderedDict()
    

    # Iterate over classifiers
    for model_name, clf in _models_.items():
        if verbose: 
            f.write("-" * 120 + "\n" + "Model: " + model_name + " Classifier\n")
            #print("-" * 120 + "\n" + "Model: " + model_name + " Classifier")
        imputer_results = OrderedDict()

        # Mean imputation is in a dictionary and iterating over that.In our case, only mean_imputation is implemented
        for imputer_name, dataframes_list in _imputers_.items():
            features_df, labels_df = split_features_labels(dataframes_list)

            years = OrderedDict()

            for df_index in range(len(dataframes_list)):
                #if verbose: print('\t\tDataset: ' + '\033[1m' + str(df_index + 1) + 'year' + '\033[0m')
                # Running K-fold cross validation on train and test set
                X_train_list, y_train_list, X_test_list, y_test_list = kfold_cv(k_folds, features_df[df_index],
                                                                                labels_df[df_index], verbose)

                metrics = OrderedDict()

                # Calculating accuracy, precision, recall, and confusion matrix
                # Initializing these variables with a numpy array of 0

                accuracy_list = np.zeros([k_folds])
                precision_list = np.zeros([k_folds, 2])
                recall_list = np.zeros([k_folds, 2])
                true_negs = np.zeros([k_folds])
                false_pos = np.zeros([k_folds])
                false_negs = np.zeros([k_folds])
                true_pos = np.zeros([k_folds])

                # Iterate over all the k-folds and calculate accuracy, precision and confusion matrix
                for k in range(k_folds):
                    X_train = X_train_list[k]
                    y_train = y_train_list[k]
                    X_test = X_test_list[k]
                    y_test = y_test_list[k]

                    # Fit the model and call predict function for test set
                    clf = clf.fit(X_train, y_train)
                    plot_importance(clf)
                    plt.show()
                    
                # Feature Importance - uncomment to run for Boosting Only
                #     a = pd.Series(dataframes[0].columns[0:64])
                #     b = pd.Series(clf.feature_importances_)
                #     df0 = pd.concat([a, b], axis=1)
                #     df0.sort_values(by=df0.columns[1])
                #     print(tabulate(df0, headers='keys', tablefmt='psql'))

                    y_test_predicted = clf.predict_proba(X_test)
                    # print(X_test[0].size)
                    # print(confusion_matrix(y_test_predicted, y_test))

                    test = np.random.rand(1,64)
                    company_prediction = clf.predict_proba(test)


                    #f.write("PREDICTION -> %f\n" %(company_prediction))
                    #print("PREDICTION -> ",company_prediction)

                    _accuracy_ = accuracy_score(y_test, y_test_predicted, normalize=True)
                    accuracy_list[k] = _accuracy_
                    _recalls_ = recall_score(y_test, y_test_predicted, average=None)
                    recall_list[k] = _recalls_

                    # code for calculating precision
                    _precisions_ = precision_score(y_test, y_test_predicted, average=None)
                    precision_list[k] = _precisions_

                    # code for calculating confusion matrix
                    _confusion_matrix_ = confusion_matrix(y_test, y_test_predicted)
                    mlp_cm = confusion_matrix(y_test, y_test_predicted)

                    # true_negs[k] = _confusion_matrix_[0][0]
                    # false_pos[k] = _confusion_matrix_[0][1]
                    # false_negs[k] = _confusion_matrix_[1][0]
                    # true_pos[k] = _confusion_matrix_[1][1]

                metrics['Accuracy'] = np.mean(accuracy_list)
                metrics['Precisions'] = np.mean(precision_list, axis=0)
                metrics['Recalls'] = np.mean(recall_list, axis=0)
                # metrics['TN'] = np.mean(true_negs)
                # metrics['FP'] = np.mean(false_pos)
                # metrics['FN'] = np.mean(false_negs)
                # metrics['TP'] = np.mean(true_pos)

                if verbose:
                    acc = metrics['Accuracy']
                    pre = metrics['Precisions']
                    rec = metrics['Recalls']
                    f.write('\t\t\tAccuracy: %f\n' %acc)
                    # f.write('\t\t\tPrecision: %f\n' %pre)
                    # f.write('\t\t\tRecall: %f\n' %rec)
                    # print('\t\t\tAccuracy:', metrics['Accuracy'])
                    # print('\t\t\tPrecision:', metrics['Precisions'])
                    # print('\t\t\tRecall:', metrics['Recalls'])

                years[str(df_index + 1) + 'year'] = metrics

            imputer_results[imputer_name] = years

        model_results[model_name] = imputer_results

    # sns.heatmap(mlp_cm,
    #             xticklabels=['Non Bankrupt', 'Bankrupt'],
    #             yticklabels=['Non Bankrupt', 'Bankrupt'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

    return model_results

def solve():

    global df_bankruptcy

    df_bankruptcy = arff_to_dataframe()
    df_bankruptcy = arff_to_dataframe()
    set_column_names(df_bankruptcy)
    convert_datatype(df_bankruptcy)
    labels_to_binary(df_bankruptcy)
    nan_dropped_dataframes = drop_nans(df_bankruptcy, verbose=True)

    mean_imputed_dataframes = mean_imputation(df_bankruptcy)

    imputed_dict = OrderedDict()
    imputed_dict['Mean'] = mean_imputed_dataframes

    seed = 7

    # Extreme Abda Boosting Classifier
    ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=seed), n_estimators=5, random_state=seed)

    # Bagging Classifier
    bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=seed), n_estimators=5, random_state=seed)

    # Neural Network Classifier - Multi layer perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(12, 12, 12), random_state=seed)

    model = XGBClassifier()

    # Building dictionary to enable calling the following classifiers and save their results as values
    models_dictionary = OrderedDict()
    models_dictionary['XGBoost'] = model
    # models_dictionary['Bagging Tree'] = bagging
    # models_dictionary['Neural Network'] = mlp

    # ideally 5 fold cross validation yielded better results
    results = perform_data_modeling(models_dictionary, imputed_dict, verbose=True, k_folds=5)

    f.close()


if __name__ == "__main__":
    solve()