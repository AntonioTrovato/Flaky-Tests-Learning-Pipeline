<<<<<<< Updated upstream
import os
import numpy as np
import pandas
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.sklearn
import logging
from urllib.parse import urlparse

DATASET_NAME = 'flakeFlagger.csv'


def loadingDataSet(datasetname):
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)



if __name__=="__main__":
    datasetSbilanciato = False
    dataset = loadingDataSet(DATASET_NAME)
    print("DATASET CARICATO CORRETTAMENTE")
    dataset_copy = dataset.copy()

    percentuale_true = (dataset.isFlaky[dataset.isFlaky == True].count() / dataset.isFlaky.count()) * 100
    if percentuale_true < 40:
        datasetSbilanciato = True
    else:
        datasetSbilanciato = False

    if not datasetSbilanciato:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_index, test_index in split.split(X=dataset, y=dataset['isFlaky']):
            train_set = dataset.loc[train_index]
            test_set = dataset.loc[test_index]
        dataset=train_set
        print("SPLIT DEL DATASET ESEGUITO CORRETTAMENTE")

    '''
    Tolgo dal dataset le fature inutili come nameProject, testCase, id.
    Successivamente divido le lable dai campioni salvandole in un altra struttura dati.
    '''
    dataset_noLable = dataset.drop(['id', 'nameProject', 'testCase', 'isFlaky'], axis=1)
    dataset_lable = dataset['isFlaky']

    # Converto le lable da boolean in int
    dataset_lable = dataset_lable.astype(int)
    print("CONVERSIONE LABLE ESEGUITA CORRETTAMENTE")

    # Standardizzo il dataset
    sc = StandardScaler()
    X_dataset = sc.fit_transform(dataset_noLable)  # Ritorna un array numpy
    print("STANDARDIZZAZIONE ESEGUITA CORRETTAMENTE")

    # Eseguo la PCA
    pca = PCA(n_components=10)
    principalCompontent = pca.fit_transform(X_dataset)
    dataset_noLable = pandas.DataFrame(principalCompontent,
                                   columns=['Principal Component 1', 'Principal Component 2', 'Principal Component  3',
                                            'Principal Component 4', 'Principal Component 5', 'Principal Component 6',
                                            'Principal Component 7', 'Principal Component 8', 'Principal Component 9',
                                            'Principal Component 10'])
    print("PCA ESEGUITA CORRETTAMENTE")

    if datasetSbilanciato==True:
        # Bilancio il data set di testo con l'oversampling SMOTE
        sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
        dataset_noLable, dataset_lable = sm.fit_resample(dataset_noLable, dataset_lable)
        dataset_noLable, dataset_lable=dataset_noLable.to_numpy(),dataset_lable.to_numpy()
        print("BILANCIAMENTO ESEGUITO CORRETTAMENTE")

    scores={
        'accuracy':[],
        'precision':[],
        'recall':[],
        'f1':[]
    }

    with mlflow.start_run():
        skf=StratifiedKFold(n_splits=10,shuffle=False)
        for skf_train_index,skf_test_index in skf.split(X=dataset_noLable,y=dataset_lable):
            X_train,X_test=dataset_noLable[skf_train_index],dataset_noLable[skf_test_index]
            y_train,y_test=dataset_lable[skf_train_index],dataset_lable[skf_test_index]
            rf=RandomForestClassifier(criterion='entropy', n_estimators=100)
            rf.fit(X=X_train,y=y_train)
            y_predict=rf.predict(X=X_test)
            scores['accuracy'].append(accuracy_score(y_true=y_test,y_pred=y_predict))
            scores['precision'].append(precision_score(y_true=y_test,y_pred=y_predict))
            scores['recall'].append(recall_score(y_true=y_test,y_pred=y_predict))
            scores['f1'].append(f1_score(y_true=y_test,y_pred=y_predict))

        if datasetSbilanciato==True:
            print("Accuracy: %.3f"% np.mean(scores['accuracy']))
            print("Precision: %.3f" % np.mean(scores['precision']))
            print("Recall: %.3f" % np.mean(scores['recall']))
            print("F1: %.3f" % np.mean(scores['f1']))
            mlflow.log_param("Dataset sbilanciato", datasetSbilanciato)
            mlflow.log_param("criterion",'entropy')
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("Accuracy",np.mean(scores['accuracy']))
            mlflow.log_metric("Precision", np.mean(scores['precision']))
            mlflow.log_metric("Recall", np.mean(scores['recall']))
            mlflow.log_metric("F1", np.mean(scores['f1']))


        clf = RandomForestClassifier(criterion='entropy', n_estimators=100)
        clf.fit(X=dataset_noLable, y=dataset_lable)
        print("FIT ESEGUITA CORRETTAMENTE")

        if datasetSbilanciato==True:
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store!="file":
                mlflow.sklearn.log_model(clf,"model",registered_model_name='RandomForest')
            else:
                mlflow.sklearn.log_model(clf,'model')


        if not datasetSbilanciato:
            print("VALUTO LE PRESTAZIONI DEL MODELLO SUL TEST SET")
            train_set_noLable = train_set.drop(['id', 'nameProject', 'testCase', 'isFlaky'], axis=1)
            train_set_lable = train_set['isFlaky']
            train_set_lable = train_set_lable.astype(int)
            X_dataset = sc.transform(train_set_noLable)  # Ritorna un array numpy
            principalCompontent = pca.transform(X_dataset)
            train_set_noLable = pandas.DataFrame(principalCompontent,
                                       columns=['Principal Component 1', 'Principal Component 2',
                                                'Principal Component  3',
                                                'Principal Component 4', 'Principal Component 5',
                                                'Principal Component 6',
                                                'Principal Component 7', 'Principal Component 8',
                                                'Principal Component 9',
                                                'Principal Component 10'])
            predict_lable=clf.predict(X=train_set_noLable)
            print("Accuracy: %.3f"%accuracy_score(y_true=train_set_lable,y_pred=predict_lable))
            print("Precision: %.3f" % precision_score(y_true=train_set_lable, y_pred=predict_lable))
            print("Recall: %.3f" % recall_score(y_true=train_set_lable, y_pred=predict_lable))
            print("F1: %.3f" % f1_score(y_true=train_set_lable, y_pred=predict_lable))
            mlflow.log_param("Dataset sbilanciato", datasetSbilanciato)
            mlflow.log_param("criterion", 'entropy')
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("Accuracy", accuracy_score(y_true=train_set_lable,y_pred=predict_lable))
            mlflow.log_metric("Precision", precision_score(y_true=train_set_lable, y_pred=predict_lable))
            mlflow.log_metric("Recall", recall_score(y_true=train_set_lable, y_pred=predict_lable))
            mlflow.log_metric("F1", f1_score(y_true=train_set_lable, y_pred=predict_lable))
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(clf, "model", registered_model_name='RandomForest')
            else:
                mlflow.sklearn.log_model(clf, 'model')
=======
import pandas
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from MajorityVoteClassifier import MajorityVoteClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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

def get_varianza_comulativa(dataset):
    # Costruisco la matrice di covarianza
    cov_mat = np.cov(dataset.T)
    # Decompongo la matrice di covarianza in un vettore composto dagli autovalori e i corrispondenti autovalori conservati come colonne in una matrice 25x25
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1, 27), var_exp, alpha=0.5, align='center', label='Varianza Individuale')
    plt.step(range(1, 27), cum_var_exp, where='mid', label='Varianza Comulativa')
    plt.ylabel('Variance Ratio')
    plt.xlabel('Numero Componenti')
    plt.legend(loc='best')
    plt.show()

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
    print("Dimensione dataset: ",len(dataset_copy))
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False] #Rimuovo dal dataset i campioni di setup e teardown
    print("Dimensione dataset dopo la rimozione dei campioni di setup e teardown: ",len(dataset_copy))
    dataset_copy=dataset_copy.reset_index()
    dataset_copy=dataset_copy.drop(['Unnamed: 0','index'],axis=1) #Rimuovo dal dataset gli indici
    train_set,test_set=split_dataset(dataset_copy) #Divido il dataset in train e test
    print("Dimensione train set: ",len(train_set))
    print("Dimensione test set: ",len(test_set))
    train_set=train_set.drop_duplicates()
    test_set = test_set.drop_duplicates()
    print("Dimensione train set: ", len(train_set))
    print("Dimensione test set: ", len(test_set))
    #Inizio flakypipeline
    train_set_copy=train_set.copy() #Lavoro sempre su una copia del train set
    #1. Rimuovo dal dataset le feature che sono object
    train_set_copy=train_set_copy.drop(get_object_colum(train_set_copy),axis=1)
    #2. Divido le etichette dai campioni converto tutto in un array numpy
    X_train_set=train_set_copy.drop(['isFlaky'],axis=1)
    y_train_set= train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    #3. Addesto il classificatore RandomForest sul train_set
    clf=RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set,y=y_train_set)
    #4. Applico le stesse operazioni al test set e verifico le prestazioni del modello
    test_set_copy=test_set.copy()
    test_set_copy=test_set_copy.drop(get_object_colum(test_set_copy),axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    y_pred=clf.predict(X=X_test_set)
    #5. Genero la matrice di confusione e le prestazioni del modello
    print("Prestazioni senza feature engineering:")
    print_performance_model(y_true=y_test_set,y_pred=y_pred)

    '''Provo ad aumentare le prestazioni del modello applicando un operazione di feature scale'''
    #1 Provo la standardizzazione
    train_set_copy=train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    #Inserisco l'operazione di feature scale
    sc=StandardScaler()
    sc.fit(X=X_train_set)
    X_train_set=sc.transform(X=X_train_set)
    dataset_scalato = pandas.DataFrame(X_train_set)
    print(dataset_scalato.describe())
    clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set, y=y_train_set)
    #Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set=sc.transform(X=X_test_set)
    y_pred = clf.predict(X=X_test_set)
    print("Prestazioni con standardizzazione:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)

    #2 Provo la normalizzazione min-max
    train_set_copy=train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    #Inserisco l'operazione di feature scale
    norm=Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set=norm.transform(X=X_train_set)
    dataset_scalato = pandas.DataFrame(X_train_set)
    print(dataset_scalato.describe())
    clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set, y=y_train_set)
    #Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set=norm.transform(X=X_test_set)
    y_pred = clf.predict(X=X_test_set)
    print("Prestazioni con normalizzazione:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)

    '''Provo ad aumentare le prestazioni del modello applicando un operazione di feature construct'''
    train_set_copy = train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    norm = Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set = norm.transform(X=X_train_set)
    get_varianza_comulativa(X_train_set)
    pca = PCA(n_components=10)
    pca.fit(X_train_set)
    X_train_set=pca.transform(X_train_set)
    print("Components = ", pca.n_components_, ";\nTotal explained variance = ",
          round(pca.explained_variance_ratio_.sum(), 5))
    clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set, y=y_train_set)
    # Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set = norm.transform(X=X_test_set)
    X_test_set=pca.transform(X_test_set)
    y_pred = clf.predict(X=X_test_set)
    print("Prestazioni con pca a 10 componenti:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)


    '''Provo ad aumentare le prestazioni applicando la fase di feature selection con boruta'''
    #Inizio flakypipeline
    #2 Provo la normalizzazione min-max
    train_set_copy=train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    columns=X_train_set.columns
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    norm=Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set=norm.transform(X=X_train_set)
    dataset_scalato = pandas.DataFrame(X_train_set,columns=columns)
    print(dataset_scalato.describe())
    rf=RandomForestClassifier(n_jobs=-1,class_weight='balanced',max_depth=15)
    feature_selector=BorutaPy(rf,n_estimators='auto',verbose=2,random_state=1)
    feature_selector.fit(X=X_train_set,y=y_train_set)
    X_train_set=feature_selector.transform(X=X_train_set)
    dataset_scalato = pandas.DataFrame(X_train_set)
    print(dataset_scalato.describe())
    clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set, y=y_train_set)
    #Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set=norm.transform(X=X_test_set)
    X_test_set=feature_selector.transform(X=X_test_set)
    y_pred = clf.predict(X=X_test_set)
    print("Prestazioni con Boruta:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)

    '''Provo ad aumentare le prestazioni bilanciando il dataset con SMOTE'''
    train_set_copy = train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    norm = Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set = norm.transform(X=X_train_set)
    ds = pandas.DataFrame(X_train_set)
    plt.title('Dataset non bilanciato')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ds.iloc[:, 0],ds.iloc[:, 1], marker='o', c=y_train_set,
                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.show()
    sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
    X_train_set, y_train_set = sm.fit_resample(X_train_set, y_train_set)

    ds = pandas.DataFrame(X_train_set)
    plt.title('Dataset non bilanciato')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ds.iloc[:, 0], ds.iloc[:, 1], marker='o', c=y_train_set,
                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.show()
    clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(X=X_train_set, y=y_train_set)
    # Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set = norm.transform(X=X_test_set)
    y_pred = clf.predict(X=X_test_set)
    print("Prestazioni con smote:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)

    '''Provo a migliorare le prestazioni con l'ensamble'''
    train_set_copy = train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    norm = Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set = norm.transform(X=X_train_set)
    ds = pandas.DataFrame(X_train_set)
    plt.title('Dataset non bilanciato')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ds.iloc[:, 0], ds.iloc[:, 1], marker='o', c=y_train_set,
                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.show()
    sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
    X_train_set, y_train_set = sm.fit_resample(X_train_set, y_train_set)

    ds = pandas.DataFrame(X_train_set)
    plt.title('Dataset non bilanciato')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ds.iloc[:, 0], ds.iloc[:, 1], marker='o', c=y_train_set,
                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.show()
    clf1 = SVC(C=100.0, kernel='rbf')
    clf2 = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf3 = KNeighborsClassifier(algorithm='brute', metric='euclidean', n_neighbors=2, weights='uniform')
    ensemble = MajorityVoteClassifier(classifiers=[clf1, clf2, clf3])
    ensemble.fit(X=X_train_set, y=y_train_set)
    # Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set = norm.transform(X=X_test_set)
    y_pred = ensemble.predict(X=X_test_set)
    print("Prestazioni con ensemble:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)


    '''Provo a migliorare le prestazioni con MLP'''
    train_set_copy = train_set.copy()
    train_set_copy = train_set_copy.drop(get_object_colum(train_set_copy), axis=1)
    X_train_set = train_set_copy.drop(['isFlaky'], axis=1)
    y_train_set = train_set_copy['isFlaky']
    X_train_set = X_train_set.to_numpy()
    y_train_set = y_train_set.to_numpy()
    norm = Normalizer(norm='max')
    norm.fit(X=X_train_set)
    X_train_set = norm.transform(X=X_train_set)
    ds = pandas.DataFrame(X_train_set)
    plt.title('Dataset non bilanciato')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ds.iloc[:, 0], ds.iloc[:, 1], marker='o', c=y_train_set,
                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.show()
    sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
    X_train_set, y_train_set = sm.fit_resample(X_train_set, y_train_set)

    ds = pandas.DataFrame(X_train_set)
    plt.title('Dataset non bilanciato')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(ds.iloc[:, 0], ds.iloc[:, 1], marker='o', c=y_train_set,
                s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.show()
    clf=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2))
    clf.fit(X=X_train_set, y=y_train_set)
    # Valuto le prestazioni del modello
    test_set_copy = test_set.copy()
    test_set_copy = test_set_copy.drop(get_object_colum(test_set_copy), axis=1)
    X_test_set = test_set_copy.drop(['isFlaky'], axis=1).to_numpy()
    y_test_set = test_set_copy['isFlaky'].to_numpy()
    X_test_set = norm.transform(X=X_test_set)
    y_pred = clf.predict(X=X_test_set)
    print("Prestazioni con MLP:")
    print_performance_model(y_true=y_test_set, y_pred=y_pred)
>>>>>>> Stashed changes
