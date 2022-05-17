import os
import pandas
from FlakyPipeline import FlakyPipeline
from sklearn.metrics import accuracy_score,recall_score,precision_score



DATASET_NAME = 'flakeFlagger.csv'


def loadingDataSet(datasetname):
    current_directory = os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)


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
