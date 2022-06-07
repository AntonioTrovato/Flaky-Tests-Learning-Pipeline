import numpy as np
import pandas
import os

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Normalizer
from sklearn import tree

from model_test import get_object_colum

DATASET_NAME='datasetFlakyTest.csv'

def print_performance_model(y_true,y_pred):
    print("Accuracy: %.3f" % accuracy_score(y_true=y_true, y_pred=y_pred))
    print("Precision: %.3f" % precision_score(y_true=y_true, y_pred=y_pred))
    print("Recall: %.3f" % recall_score(y_true=y_true, y_pred=y_pred))
    print("F1: %.3f" % f1_score(y_true=y_true, y_pred=y_pred))


def loadingDataSet(datasetname):
    current_directory=os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)

def use_interpretable_model(dataset):

    dataset=dataset.drop_duplicates()
    dataset=dataset.reset_index()
    dataset=dataset.drop(['index'],axis=1)
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index_stratified,test_index_stratified in split.split(dataset,dataset['isFlaky']):
        train_set=dataset.loc[train_index_stratified]
        test_set=dataset.loc[test_index_stratified]


    train_set_copy = train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    columns=X_train_set.columns #salvo le colonne
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    norm = Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set = norm.transform(X=X_train_set)

    df = pandas.DataFrame(X_train_set, columns=columns)
    rf_fs = RandomForestClassifier(n_estimators=len(X_train_set), random_state=0, n_jobs=-1)
    rf_fs.fit(X=X_train_set, y=y_train_set)
    importance = rf_fs.feature_importances_
    indices = np.argsort(importance)[::-1]
    colum_remove = []
    for f in range(X_train_set.shape[1]):
        if importance[indices[f]] < 0.02:
            colum_remove.append(columns[indices[f]])
    df = df.drop(colum_remove, axis=1)
    columns=df.columns
    X_train_set=df.to_numpy()
    sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_train_set, y_train_set = sm.fit_resample(X_train_set, y_train_set)
    X_train_set=pandas.DataFrame(X_train_set,columns=columns)
    #come modello surrogato globale abbiamo scelto un albero di decisione
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    rf_clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set, y=y_train_set)
    rf_clf.fit(X=X_train_set, y=y_train_set)


    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)

    X_test_set = test_set_copy.drop(['isFlaky'], axis=1)
    full_colums = X_test_set.columns
    X_test_set=X_test_set.to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set = norm.transform(X=X_test_set)
    df=pandas.DataFrame(X_test_set,columns=full_colums)
    df=df.drop(colum_remove,axis=1)
    X_test_set=df
    y_pred = clf.predict(X=X_test_set)
    rfy_pred = rf_clf.predict(X=X_test_set)
    print("PRESTAZIONI MODELLO SURROGATO GLOBALE NEL TEST:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)

    #calcoliamo R^2 per verificare la bontà del surrogato
    print("R^2 SCORE: ")
    print(r2_score(rfy_pred,y_pred))
    tree.plot_tree(clf)
    plt.show()
    plt.savefig('Interpretabiliti.png', dpi=600)