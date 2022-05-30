import os
import pandas
from FlakyPipeline import FlakyPipeline
<<<<<<< Updated upstream
from sklearn.metrics import accuracy_score,recall_score,precision_score
=======
>>>>>>> Stashed changes


DATASET_NAME = 'datasetFlakyTest.csv'

<<<<<<< Updated upstream
DATASET_NAME = 'flakeFlagger.csv'

=======
>>>>>>> Stashed changes

def loadingDataSet(datasetname):
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)

if __name__=="__main__":
    dataset = loadingDataSet(DATASET_NAME)  # Carico il dataset
    dataset_copy = dataset.copy()  # Lavoro su una copia del dataset
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    dataset_copy = dataset_copy.reset_index()
    dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
    flakyPipeline=FlakyPipeline()
    flakyPipeline.fit(dataset_copy)

<<<<<<< Updated upstream
dataset=loadingDataSet(DATASET_NAME)
flakyPipeline=FlakyPipeline()
flakyPipeline.fit(dataset)

#dataset_noLable = dataset.drop(['isFlaky'], axis=1)
#dataset_lable = dataset['isFlaky']
#predict=flakyPipeline.predict(dataset_noLable)


#dataset_lable = dataset_lable.astype(int)
#print("Accuracy: %.3f" % accuracy_score(y_true=dataset_lable, y_pred=predict))
#print("Precision: %.3f" % precision_score(y_true=dataset_lable, y_pred=predict))
#print("Recall: %.3f" % recall_score(y_true=dataset_lable, y_pred=predict))
=======

>>>>>>> Stashed changes
