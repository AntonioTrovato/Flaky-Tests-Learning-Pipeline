import os
import pandas
import numpy as np
import random
from FlakyPipeline import FlakyPipeline


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
print(dataset_copy.info())