import os
import pandas
from FlakyPipeline import FlakyPipeline


DATASET_NAME = 'datasetFlakyTest.csv'


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


