import pandas
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from MajorityVoteClassifier import MajorityVoteClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression
from SBS import SBS
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

DATASET_NAME='datasetFlakyTest.csv'

def loadingDataSet(datasetname):
    current_directory=os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)

def split_dataset(dataset):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index_stratified, test_index_stratified in split.split(dataset, dataset['isFlaky']):
        train_set = dataset.loc[train_index_stratified]
        test_set = dataset.loc[test_index_stratified]
    return train_set,test_set

def get_object_colum(dataset):
    drop_col = []
    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            drop_col.append(col)
    return drop_col



def print_performance_model(y_true,y_pred):
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predict label')
    plt.ylabel('true label')
    plt.show()
    print("Accuracy: %.3f" % accuracy_score(y_true=y_true, y_pred=y_pred))
    print("Precision: %.3f" % precision_score(y_true=y_true, y_pred=y_pred))
    print("Recall: %.3f" % recall_score(y_true=y_true, y_pred=y_pred))
    print("F1: %.3f" % f1_score(y_true=y_true, y_pred=y_pred))


if __name__=="__main__":
    dataset=loadingDataSet(DATASET_NAME) #Carico il dataset
    dataset_copy=dataset.copy()  #Lavoro su una copia del dataset
    dataset_copy=dataset_copy.drop_duplicates()#Rimuovo i duplicati
    print("Dimensione dataset: ",len(dataset_copy))
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False] #Rimuovo dal dataset i campioni di setup e teardown
    print("Dimensione dataset dopo la rimozione dei campioni di setup e teardown: ",len(dataset_copy))
    dataset_copy=dataset_copy.reset_index()
    dataset_copy=dataset_copy.drop(['Unnamed: 0','index'],axis=1) #Rimuovo dal dataset gli indici
    train_set,test_set=split_dataset(dataset_copy) #Divido il dataset in train e test
    print("Dimensione train set: ",len(train_set))
    print("Dimensione test set: ",len(test_set))


    #Inizio flakypipeline
    train_set_copy=train_set.copy() #Lavoro sempre su una copia del train set
    #1. Rimuovo dal dataset le feature che sono object
    train_set_copy=train_set_copy.drop(get_object_colum(train_set_copy),axis=1)
    #2. Divido le etichette dai campioni converto tutto in un array numpy
    X_train_set=train_set_copy.drop(['isFlaky'],axis=1)
    y_train_set= train_set_copy['isFlaky']
    columns=X_train_set.columns #Salvo il nome delle colonne
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
    X_train_set = df.to_numpy()
    sm = SMOTE(sampling_strategy='all', k_neighbors=3, random_state=42)
    X_train_set, y_train_set = sm.fit_resample(X_train_set, y_train_set)
    #3. Addesto il classificatore RandomForest sul train_set
    clf = MLPClassifier(max_iter=100,activation='relu',alpha=0.0001,hidden_layer_sizes=(50,100,50),learning_rate='constant',solver='adam')
    clf.fit(X=X_train_set, y=y_train_set)
    y_pred = clf.predict(X_train_set)
    print("Prestazioni train set")
    print_performance_model(y_true=y_train_set, y_pred=y_pred)
    # 4. Applico le stesse operazioni al test set e verifico le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set = norm.transform(X=X_test_set)
    df = pandas.DataFrame(X_test_set, columns=columns)
    df = df.drop(colum_remove, axis=1)
    X_test_set = df.to_numpy()
    y_pred = clf.predict(X=X_test_set)
    # 5. Genero la matrice di confusione e le prestazioni del modello
    print("Prestazioni con MLP:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)








    '''
    mlp = MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3,scoring='f1_micro')
    clf.fit(X_train_set, y_train_set)
    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    '''
