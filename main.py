from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression


DATASET_NAME='flakeFlagger.csv'

def loadingDataSet(datasetname):
    current_directory=os.getcwd()
    csv_path = os.path.join(current_directory, datasetname)
    return pandas.read_csv(csv_path)


#Partiziono il dataset utilizzando un campionamento statificato
dataset=loadingDataSet(DATASET_NAME)
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index_stratified,test_index_stratified in split.split(dataset,dataset['isFlaky']):
    train_set=dataset.loc[train_index_stratified]
    test_set=dataset.loc[test_index_stratified]


train_set_copy=train_set.copy()

'''
Tolgo dal dataset le fature inutili come nameProject, testCase, id.
Successivamente divido le lable dai campioni salvandole in un altra struttura dati.
'''
train_dataset=train_set.drop(['id','nameProject','testCase','isFlaky'],axis=1)
train_dataset_lable=train_set['isFlaky']


#Converto le lable da boolean in int
train_dataset_lable=train_dataset_lable.astype(int)

#Standardizzo il dataset
sc=StandardScaler()
X_train_dataset=sc.fit_transform(train_dataset)


'''
Utilizzo la pca per eseguire la riduzione della dimensionalità, tuttavia la pca sfrutta la varianza,
pertanto mi occorre sapere quante componenti dovrei mantenere per conseravare almento l'80% di varianza
'''
#Verifico quante caratteristiche occorrono per una pca con l 80% di varianza
#Costruisco la matrice di covarianza
cov_mat=np.cov(X_train_dataset.T) #Calcolo la matrice di covarianza del dataset di addestramento standardizzato
#Decompongo la matrice di covarianza in un vettore composto dagli autovalori e i corrispondenti autovalori conservati come colonne in una matrice 25x25

eigen_vals, eigen_vecs=np.linalg.eig(cov_mat)
tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
plt.bar(range(1,26),var_exp,alpha=0.5,align='center',label='Varianza Individuale')
plt.step(range(1,26),cum_var_exp,where='mid',label='Varianza Comulativa')
plt.ylabel('Variance Ratio')
plt.xlabel('Numero Componenti')
plt.legend(loc='best')
plt.show()

#Eseguo la PCA
pca=PCA(n_components=10)
principalCompontent=pca.fit_transform(X_train_dataset)
pca_train_dataset=pandas.DataFrame(principalCompontent,columns=['Principal Component 1','Principal Component 2','Principal Component  3',
                                                                'Principal Component 4','Principal Component 5','Principal Component 6',
                                                                'Principal Component 7','Principal Component 8','Principal Component 9',
                                                                'Principal Component 10'])




#Bilancio il data set di testo con l'oversampling SMOTE
plt.title('Dataset non bilanciato')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(pca_train_dataset.iloc[:, 0], pca_train_dataset.iloc[:, 1], marker='o', c=train_dataset_lable,
           s=25, edgecolor='k', cmap=plt.cm.coolwarm)
plt.show()


sm=SMOTE(sampling_strategy='auto', k_neighbors=1,random_state=42)
X_train_dataset,Y_train_dataset=sm.fit_resample(pca_train_dataset,train_dataset_lable)

plt.title('Dataset bilanciato con SMOTE')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X_train_dataset.iloc[:, 0], X_train_dataset.iloc[:, 1], marker='o', c=Y_train_dataset,
           s=25, edgecolor='k', cmap=plt.cm.coolwarm)
plt.show()

#Provo perceptron
train_sizes, train_scores, test_scores =learning_curve(estimator=Perceptron(),
                                                       X=X_train_dataset,
                                                       y=Y_train_dataset,
                                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                                       cv=10,
                                                       n_jobs=1)



train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samplel')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Perceptron Accuracy")
plt.ylim([0.0,1.0])
plt.show()

#Provo la regressione logica
train_sizes, train_scores, test_scores =learning_curve(estimator=LogisticRegression(),
                                                       X=X_train_dataset,
                                                       y=Y_train_dataset,
                                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                                       cv=10,
                                                       n_jobs=1)



train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samplel')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Logistic Regression Accuracy")
plt.ylim([0.0,1.0])
plt.show()


#Cross validation decision tree
train_sizes, train_scores, test_scores =learning_curve(estimator=DecisionTreeClassifier(),
                                                       X=X_train_dataset,
                                                       y=Y_train_dataset,
                                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                                       cv=10,
                                                       n_jobs=1)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samplel')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Decision-Tree Accuracy")
plt.ylim([0.0,1.0])
plt.show()


#Cross validation random forest
train_sizes, train_scores, test_scores =learning_curve(estimator=RandomForestClassifier(n_estimators=15),
                                                       X=X_train_dataset,
                                                       y=Y_train_dataset,
                                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                                       cv=10,
                                                       n_jobs=1)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samplel')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Random Forest Accuracy")
plt.ylim([0.0,1.0])
plt.show()



#Cross validation SVM

train_sizes, train_scores, test_scores =learning_curve(estimator=SVC(kernel='rbf',C=1.0,random_state=0, gamma=2),
                                                       X=X_train_dataset,
                                                       y=Y_train_dataset,
                                                       cv=10,
                                                       n_jobs=1)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samplel')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("SVM Accuracy")
plt.ylim([0.0,1.0])
plt.show()


'''Il modello che si adatta meglio al dateset risulta essere il random forest, eseguo il tuning degli iperparametri'''


#TEST
test_set_copy=test_set.copy()

'''
Tolgo dal dataset le fature inutili come nameProject, testCase, id.
Successivamente divido le lable dai campioni salvandole in un altra struttura dati.
'''
test_dataset=test_set.drop(['id','nameProject','testCase','isFlaky'],axis=1)
test_dataset_lable=test_set['isFlaky']
print(test_dataset_lable.describe())
test_dataset_lable=test_dataset_lable.astype(int)


X_test_dataset=sc.transform(test_dataset)
principalCompontent=pca.transform(X_test_dataset)
pca_test_dataset=pandas.DataFrame(principalCompontent,columns=['Principal Component 1','Principal Component 2','Principal Component  3',
                                                                'Principal Component 4','Principal Component 5','Principal Component 6',
                                                                'Principal Component 7','Principal Component 8','Principal Component 9',

                                                                'Principal Component 10'])
randomForest=RandomForestClassifier(n_estimators=40,class_weight='balanced_subsample')
randomForest.fit(X_train_dataset,Y_train_dataset)
test_predict=randomForest.predict(pca_test_dataset)
confmat=confusion_matrix(y_true=test_dataset_lable,y_pred=test_predict)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.1)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predict label')
plt.ylabel('true label')
plt.show()

print("Accuracy: %.3f" %accuracy_score(y_true=test_dataset_lable,y_pred=test_predict))
print("Precision: %.3f" %precision_score(y_true=test_dataset_lable,y_pred=test_predict))
print("Recall: %.3f" %recall_score(y_true=test_dataset_lable,y_pred=test_predict))
print("F1: %.3f" %f1_score(y_true=test_dataset_lable,y_pred=test_predict))
