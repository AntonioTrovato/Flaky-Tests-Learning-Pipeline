import os
import pandas
import numpy as np
import random
from FlakyPipeline import FlakyPipeline


'''
current_directory = os.getcwd()
csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
dataset = pandas.read_csv(csv_path)
dataset_copy = dataset
print(dataset_copy.info())
is_flaky = dataset_copy['isFlaky'] == True
is_flaky_row=dataset_copy[is_flaky]
for index in is_flaky_row['Unnamed: 0']:
    is_flaky_row.at[index,'isFlaky']=random.randint(0, 1)
dataset_copy=dataset_copy.append([is_flaky_row]*7,ignore_index=True)
dataset_copy=dataset_copy.drop('Unnamed: 0', axis=1)
dataset_copy=dataset_copy.reset_index()
dataset_copy.rename(columns={'index':'Unnamed: 0'},inplace=True)
print(dataset_copy.info())


current_directory = os.getcwd()
csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
dataset = pandas.read_csv(csv_path)
dataset_copy = dataset
dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
dataset_copy = dataset_copy.reset_index()
dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
columns=list(dataset_copy.columns)
columns.remove('isFlaky')
random.shuffle(columns)
dataset_copy=dataset_copy.drop([columns[0],columns[1],columns[2],columns[4]],axis=1)
'''

current_directory = os.getcwd()
csv_path = os.path.join(current_directory, 'datasetFlakyTest.csv')
dataset = pandas.read_csv(csv_path)
dataset_copy = dataset
dataset_copy = dataset_copy[dataset_copy['testCase'].str.lower().str.contains('.setup|.teardown') == False]  # Rimuovo dal dataset i campioni di setup e teardown
dataset_copy = dataset_copy.reset_index()
dataset_copy = dataset_copy.drop(['Unnamed: 0', 'index'], axis=1)  # Rimuovo dal dataset gli indici
dataset_copy=dataset_copy.drop_duplicates()

dataset_noLable=dataset_copy.drop(['isFlaky'], axis=1)
dataset_noLable=dataset_noLable.drop_duplicates()
dataset_lable=dataset_copy['isFlaky']
print(len(dataset_lable))
print(len(dataset_noLable))