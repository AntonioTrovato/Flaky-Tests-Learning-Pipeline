import numpy as np
import pandas
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import pickle


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

        with mlflow.start_run(experiment_id='0'):
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
            return y_pred



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
