import os
import pandas
import random
import numpy as np
from FlakyPipeline import FlakyPipeline



def setup_module():
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
    dataset_copy = pandas.read_csv(csv_path).copy()
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    dataset_copy = dataset_copy.reset_index()
    dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
    return dataset_copy



def test_fit_permutation_features():
    '''
    Input: Dataset con permutazioni delle colonne
    Oracolo: La pipeline garantisce le stesse prestazioni
    '''

    dataset=setup_module()
    columns=list(dataset.columns)
    random.shuffle(columns)
    dataset=dataset.loc[:,columns]
    flakyPipeline = FlakyPipeline()
    flakyPipeline.fit(dataset)
    accuracy=int(flakyPipeline.__getattribute__('accuracy')*100)
    precision=int(flakyPipeline.__getattribute__('precision')*100)
    recall=int(flakyPipeline.__getattribute__('recall')*100)
    f1=int(flakyPipeline.__getattribute__('f1')*100)
    assert accuracy in range(92,100) and precision in range(59,67) and recall in range(68,74) and f1 in range(62,70)


def test_fit_add_redundant_date():
    '''Testo le prestazioni della pipeline passando il dataset con una aggiunta di dati ridondanti
        Oracolo: La pipeline garantisce le stesse prestazioni
    '''
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
    dataset = pandas.read_csv(csv_path)
    dataset_copy = dataset
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    dataset_copy = dataset_copy.reset_index()
    dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
    is_flaky = dataset_copy['isFlaky'] == True
    is_flaky_row = dataset_copy[is_flaky]
    dataset_copy = dataset_copy.append([is_flaky_row] * 20, ignore_index=True)
    flakyPipeline = FlakyPipeline()
    flakyPipeline.fit(dataset_copy)
    accuracy = int(flakyPipeline.__getattribute__('accuracy') * 100)
    precision = int(flakyPipeline.__getattribute__('precision') * 100)
    recall = int(flakyPipeline.__getattribute__('recall') * 100)
    f1 = int(flakyPipeline.__getattribute__('f1') * 100)
    assert accuracy in range(92,100) and precision in range(57,66) and recall in range(64,71) and f1 in range(60,67)

def test_fit_add_false_data():
    '''Testo le prestazioni della pipeline passando il dataset con una aggiunta di dati rietichettati a caso
        Oracolo: La pipeline restituisce prestazioni inferiori
    '''
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
    dataset = pandas.read_csv(csv_path)
    dataset_copy = dataset
    is_flaky = dataset_copy['isFlaky'] == True
    is_flaky_row = dataset_copy[is_flaky]
    for index in is_flaky_row['Unnamed: 0']:
        is_flaky_row.at[index, 'isFlaky'] = random.randint(0, 1)
    dataset_copy = dataset_copy.append([is_flaky_row] * 7, ignore_index=True)
    dataset_copy = dataset_copy.drop('Unnamed: 0', axis=1)
    flakyPipeline = FlakyPipeline()
    flakyPipeline.fit(dataset_copy)
    accuracy = int(flakyPipeline.__getattribute__('accuracy') * 100)
    precision = int(flakyPipeline.__getattribute__('precision') * 100)
    recall = int(flakyPipeline.__getattribute__('recall') * 100)
    f1 = int(flakyPipeline.__getattribute__('f1') * 100)
    assert accuracy < 92 and precision <57 and recall <64 and f1 <60

def test_fit_remove_data():
    '''Testo le prestazioni della pipeline passando il dataset con campioni random rimossi
        Oracolo: La pipeline restituisce le stesse prestazioni
    '''
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
    dataset = pandas.read_csv(csv_path)
    dataset_copy = dataset
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains(
        '.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    dataset_copy = dataset_copy.reset_index()
    dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
    np.random.seed(10)
    remove_n = 5000
    drop_indices = np.random.choice(dataset_copy.index, remove_n, replace=False)
    dataset_subset = dataset_copy.drop(drop_indices)
    flakyPipeline = FlakyPipeline()
    flakyPipeline.fit(dataset_subset)
    accuracy = int(flakyPipeline.__getattribute__('accuracy') * 100)
    precision = int(flakyPipeline.__getattribute__('precision') * 100)
    recall = int(flakyPipeline.__getattribute__('recall') * 100)
    f1 = int(flakyPipeline.__getattribute__('f1') * 100)
    assert accuracy in range(92,100) and precision in range(57,66) and recall in range(64,71) and f1 in range(60,67)

def test_fit_remove_features():
    '''Testo le prestazioni della pipeline passando il dataset con features random rimosse
        Oracolo: La pipeline restituisce prestazioni inferiori
    '''
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
    dataset = pandas.read_csv(csv_path)
    dataset_copy = dataset
    dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
    dataset_copy = dataset_copy.reset_index()
    dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
    columns = list(dataset_copy.columns)
    columns.remove('isFlaky')
    random.shuffle(columns)
    dataset_copy = dataset_copy.drop([columns[0], columns[1], columns[2], columns[4],columns[5],columns[6]], axis=1)
    flakyPipeline = FlakyPipeline()
    flakyPipeline.fit(dataset_copy)
    accuracy = int(flakyPipeline.__getattribute__('accuracy') * 100)
    precision = int(flakyPipeline.__getattribute__('precision') * 100)
    recall = int(flakyPipeline.__getattribute__('recall') * 100)
    f1 = int(flakyPipeline.__getattribute__('f1') * 100)
    assert accuracy < 92 and precision <57 and recall <64 and f1 <60