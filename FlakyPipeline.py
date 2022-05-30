import numpy as np
import pandas
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
<<<<<<< Updated upstream
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedKFold
=======
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
>>>>>>> Stashed changes
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import pickle

<<<<<<< Updated upstream
class FlakyPipeline:

    def __init__(self):
         self.std=None
         self.pca=None
         self.smote=None
         self.classificatore = None
         self.datasetSbilanciato=False


    def fit(self,X):


        X_copy = X.copy()
        percentuale_true = (X.isFlaky[X.isFlaky == True].count() / X.isFlaky.count()) * 100 #Calcolo se il dataset è sbilanciato
        if percentuale_true < 40:
            datasetSbilanciato = True
        else:
            datasetSbilanciato = False

        if not datasetSbilanciato:
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42) #Eseguo uno split del dataset
            for train_index, test_index in split.split(X=X, y=X['isFlaky']):
                train_set = X.loc[train_index]
                test_set = X.loc[test_index]
            dataset = train_set
            print("SPLIT DEL DATASET ESEGUITO CORRETTAMENTE")

        '''
        Tolgo dal dataset le fature inutili come nameProject, testCase, id.
        Successivamente divido le lable dai campioni salvandole in un altra struttura dati.
        '''
        dataset_noLable = X.drop(['id', 'nameProject', 'testCase', 'isFlaky'], axis=1)
        dataset_lable = X['isFlaky']

        # Converto le lable da boolean in int
        dataset_lable = dataset_lable.astype(int)
        print("CONVERSIONE LABLE ESEGUITA CORRETTAMENTE")

        # Standardizzo il dataset
        self.std = StandardScaler()
        X_dataset = self.std.fit_transform(dataset_noLable)  # Ritorna un array numpy
        print("STANDARDIZZAZIONE ESEGUITA CORRETTAMENTE")

        # Eseguo la PCA
        self.pca = PCA(n_components=10)
        principalCompontent = self.pca.fit_transform(X_dataset)
        dataset_noLable = pandas.DataFrame(principalCompontent,
                                           columns=['Principal Component 1', 'Principal Component 2',
                                                    'Principal Component  3',
                                                    'Principal Component 4', 'Principal Component 5',
                                                    'Principal Component 6',
                                                    'Principal Component 7', 'Principal Component 8',
                                                    'Principal Component 9',
                                                    'Principal Component 10'])
        print("PCA ESEGUITA CORRETTAMENTE")

        if datasetSbilanciato == True:
            # Bilancio il data set di testo con l'oversampling SMOTE
            self.smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
            dataset_noLable, dataset_lable = self.smote.fit_resample(dataset_noLable, dataset_lable)
            dataset_noLable, dataset_lable = dataset_noLable.to_numpy(), dataset_lable.to_numpy()
            print("BILANCIAMENTO ESEGUITO CORRETTAMENTE")

        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        with mlflow.start_run():
            skf = StratifiedKFold(n_splits=10, shuffle=False)
            for skf_train_index, skf_test_index in skf.split(X=dataset_noLable, y=dataset_lable):
                X_train, X_test = dataset_noLable[skf_train_index], dataset_noLable[skf_test_index]
                y_train, y_test = dataset_lable[skf_train_index], dataset_lable[skf_test_index]
                rf = RandomForestClassifier(criterion='entropy', n_estimators=100)
                rf.fit(X=X_train, y=y_train)
                y_predict = rf.predict(X=X_test)
                scores['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_predict))
                scores['precision'].append(precision_score(y_true=y_test, y_pred=y_predict))
                scores['recall'].append(recall_score(y_true=y_test, y_pred=y_predict))
                scores['f1'].append(f1_score(y_true=y_test, y_pred=y_predict))

            if datasetSbilanciato == True:
                print("Accuracy: %.3f" % np.mean(scores['accuracy']))
                print("Precision: %.3f" % np.mean(scores['precision']))
                print("Recall: %.3f" % np.mean(scores['recall']))
                print("F1: %.3f" % np.mean(scores['f1']))
                mlflow.log_param("Dataset sbilanciato", datasetSbilanciato)
                mlflow.log_param("criterion", 'entropy')
                mlflow.log_param("n_estimators", 100)
                mlflow.log_metric("Accuracy", np.mean(scores['accuracy']))
                mlflow.log_metric("Precision", np.mean(scores['precision']))
                mlflow.log_metric("Recall", np.mean(scores['recall']))
                mlflow.log_metric("F1", np.mean(scores['f1']))

            self.classificatore = RandomForestClassifier(criterion='entropy', n_estimators=100)
            self.classificatore.fit(X=dataset_noLable, y=dataset_lable)
            print("FIT ESEGUITA CORRETTAMENTE")

            if datasetSbilanciato == True:
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(self.classificatore, "model", registered_model_name='RandomForest')
                else:
                    mlflow.sklearn.log_model(self.classificatore, 'model')

            if not datasetSbilanciato:
                print("VALUTO LE PRESTAZIONI DEL MODELLO SUL TEST SET")
                train_set_noLable = train_set.drop(['id', 'nameProject', 'testCase', 'isFlaky'], axis=1)
                train_set_lable = train_set['isFlaky']
                train_set_lable = train_set_lable.astype(int)
                X_dataset = self.std.transform(train_set_noLable)  # Ritorna un array numpy
                principalCompontent = self.pca.transform(X_dataset)
                train_set_noLable = pandas.DataFrame(principalCompontent,
                                                     columns=['Principal Component 1', 'Principal Component 2',
                                                              'Principal Component  3',
                                                              'Principal Component 4', 'Principal Component 5',
                                                              'Principal Component 6',
                                                              'Principal Component 7', 'Principal Component 8',
                                                              'Principal Component 9',
                                                              'Principal Component 10'])
                predict_lable =self.classificatore.predict(X=train_set_noLable)
                print("Accuracy: %.3f" % accuracy_score(y_true=train_set_lable, y_pred=predict_lable))
                print("Precision: %.3f" % precision_score(y_true=train_set_lable, y_pred=predict_lable))
                print("Recall: %.3f" % recall_score(y_true=train_set_lable, y_pred=predict_lable))
                print("F1: %.3f" % f1_score(y_true=train_set_lable, y_pred=predict_lable))
                mlflow.log_param("Dataset sbilanciato", datasetSbilanciato)
                mlflow.log_param("criterion", 'entropy')
                mlflow.log_param("n_estimators", 100)
                mlflow.log_metric("Accuracy", accuracy_score(y_true=train_set_lable, y_pred=predict_lable))
                mlflow.log_metric("Precision", precision_score(y_true=train_set_lable, y_pred=predict_lable))
                mlflow.log_metric("Recall", recall_score(y_true=train_set_lable, y_pred=predict_lable))
                mlflow.log_metric("F1", f1_score(y_true=train_set_lable, y_pred=predict_lable))
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(self.classificatore, "model", registered_model_name='RandomForest')
                else:
                    mlflow.sklearn.log_model(self.classificatore, 'model')

        #Save STD,PCA and Model to disk
        pickle.dump(self.std,open('finalized_std.sav','wb'))
        pickle.dump(self.pca, open('finalized_pca.sav', 'wb'))
        pickle.dump(self.classificatore, open('finalized_model.sav', 'wb'))
        print("STD, PCA E MODEL SALVATI")


    def predict(self,X):
        load_std=pickle.load(open('finalized_std.sav', 'rb'))
        load_pca=pickle.load(open('finalized_pca.sav', 'rb'))
        load_model=pickle.load(open('finalized_model.sav', 'rb'))
        if load_std is None or load_pca is None or load_model is None:
            print("ESEGUIRE PRIMA LA FIT")
        else:
            if 'id' in X:
                X=X.drop(['id'],axis=1)
            if 'nameProject' in X:
                X=X.drop(['nameProject'],axis=1)
            if 'testCase' in X:
                X = X.drop(['testCase'], axis=1)
            X=load_std.transform(X)
            X=load_pca.transform(X)
            y_pred=load_model.predict(X)
=======

class FlakyPipeline:

    def __init__(self):
        self.normalizer = None
        self.classificatore = None
        self.accuracy = None
        self.precision= None
        self.recall=None
        self.f1=None


    def fit(self, X):
        # Partizionamento dataset
        train_set, test_set = self.__Split(X=X) #Eseguo lo split del dataset

        '''Inizio Pipeline'''

        #Fase data clean
        train_set = train_set.drop_duplicates()
        test_set = test_set.drop_duplicates()
        train_set = train_set.loc[:, ~train_set.columns.duplicated()]  # Rimuovo colonne duplicate
        train_set = train_set.drop(self.__get_object_colum(train_set), axis=1)  # Rimuovo colonne di tipo object
        X_train_set = train_set.drop(['isFlaky'], axis=1)
        y_train_set = train_set['isFlaky']


        columns=X_train_set.columns #Salvo le colonne prima di trasformare il dataframe in un array numpy
        X_train_set = X_train_set.to_numpy()
        y_train_set = y_train_set.to_numpy()

        #Fase di feature scale
        X_train_set = self.__Normalization(X=X_train_set)

        #Fase di feature selection
        X_train_set,drop_col=self.__FeatureSelection(X=X_train_set,y=y_train_set,columns=columns)

        #Fase di data balancing
        X_train_set, y_train_set = self.__SMOTE(X=X_train_set, y=y_train_set)

        with mlflow.start_run(experiment_id=0):
            self.classificatore = RandomForestClassifier(criterion='entropy', n_estimators=150)
            self.classificatore.fit(X=X_train_set, y=y_train_set)
            print("FIT ESEGUITA CORRETTAMENTE")
            # Save Norm and Model to disk
            pickle.dump(self.normalizer, open('finalized_norm.sav', 'wb'))
            pickle.dump(self.classificatore, open('finalized_model.sav', 'wb'))
            pickle.dump(drop_col, open('finalized_drop_col.sav', 'wb'))
            print("NORM, FEATURE SELECT E MODEL SALVATI")

            X_test_set = test_set.drop(['isFlaky'], axis=1)
            y_test_set = test_set['isFlaky']
            y_test_set = y_test_set.to_numpy()

            predict_lable = self.predict(X=X_test_set)
            print("PRESTAZIONI SU TEST SET")
            self.__getReport(accuracy=accuracy_score(y_true=y_test_set, y_pred=predict_lable),
                             precision=precision_score(y_true=y_test_set, y_pred=predict_lable),
                             recall=recall_score(y_true=y_test_set, y_pred=predict_lable),
                             f1=f1_score(y_true=y_test_set, y_pred=predict_lable))

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(self.classificatore, "model", registered_model_name='RandomForest')
            else:
                mlflow.sklearn.log_model(self.classificatore, 'model')

    def predict(self, X):
        load_norm = pickle.load(open('finalized_norm.sav', 'rb'))
        load_model = pickle.load(open('finalized_model.sav', 'rb'))
        load_drop_col= pickle.load(open('finalized_drop_col.sav', 'rb'))
        if load_norm is None or load_model is None or load_drop_col is None:
            print("ESEGUIRE PRIMA LA FIT")
        else:
            # Fase di data cleaning
            X = X.loc[:, ~X.columns.duplicated()]  # Rimuovo colonne duplicate
            X = X.drop(self.__get_object_colum(X), axis=1)
            colums=X.columns
            X = X.to_numpy()

            # Fase di feature scale
            X = load_norm.transform(X=X)

            # Fase di feature selection
            df=pandas.DataFrame(X,columns=colums)
            df=df.drop(load_drop_col,axis=1)
            X=df.to_numpy()
            y_pred = load_model.predict(X)
>>>>>>> Stashed changes
            return y_pred



<<<<<<< Updated upstream
=======
    def __Split(self, X):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Eseguo uno split del dataset
        for train_index, test_index in split.split(X=X, y=X['isFlaky']):
            train_set = X.loc[train_index]
            test_set = X.loc[test_index]
        print("SPLIT DEL DATASET ESEGUITO CORRETTAMENTE")
        return train_set, test_set

    def __get_object_colum(self, X):
        drop_col = []
        for col in X.columns:
            if X[col].dtypes == 'object':
                drop_col.append(col)
        return drop_col

    def __Normalization(self, X):
        self.normalizer = Normalizer(norm='max')
        self.normalizer.fit(X=X)
        X = self.normalizer.transform(X)  # Ritorna un array numpy
        print("NORMALIZZAZIONE ESEGUITA CORRETTAMENTE")
        return X

    def __FeatureSelection(self,X,y,columns):
        df = pandas.DataFrame(X, columns=columns)
        rf_fs = RandomForestClassifier(n_estimators=len(X), random_state=0, n_jobs=-1)
        rf_fs.fit(X=X, y=y)
        importance = rf_fs.feature_importances_
        indices = np.argsort(importance)[::-1]
        colum_remove = []
        for f in range(X.shape[1]):
            if importance[indices[f]] < 0.02:
                colum_remove.append(columns[indices[f]])
        df = df.drop(colum_remove, axis=1)
        X = df.to_numpy()
        return X,colum_remove


    def __SMOTE(self, X, y):
        # Bilancio il data set di testo con l'oversampling SMOTE
        self.smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        X, y = self.smote.fit_resample(X, y)
        print("BILANCIAMENTO ESEGUITO CORRETTAMENTE")
        return X, y

    def __getReport(self, accuracy, precision, recall, f1):
        self.accuracy=accuracy
        self.precision=precision
        self.recall=recall
        self.f1=f1
        print("Accuracy: %.3f" % accuracy)
        print("Precision: %.3f" % precision)
        print("Recall: %.3f" % recall)
        print("F1: %.3f" % f1)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        dict_param = self.classificatore.get_params(deep=True)
        for parm in dict_param:
            mlflow.log_param(parm, dict_param.get(parm))
>>>>>>> Stashed changes
