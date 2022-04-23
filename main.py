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