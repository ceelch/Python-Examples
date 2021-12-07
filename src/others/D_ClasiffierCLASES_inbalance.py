#%%
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)

from collections import Counter

from sklearn.feature_selection import mutual_info_classif

#%%
def getAUC(p,c,c0):
    '''
    p: probabilidad de la clase, p(C|x)
    c: array de clases reales (c,x)
    c0: la clase (positiva) para la que calculamos el AUC (int)
    '''    
    c01= np.array(c!=c0,dtype=np.int)
    p0= p[:,c0]
    
    (N0,N1)= np.unique(c01,return_counts=True)[1]
    delta= N0
    tau= 0
    
    sort= np.argsort(-p0)
    for i in range(len(sort)):
        if(c01[sort[i]]==0):#cuando es un cero
            delta-=1
        else:#cuando es un 1
            tau+=delta
    AUC= 1.0 - (tau/(N0*N1))    
    return AUC

def getAvgAUC(p,c):
    return np.average([getAUC(p,c,c0) for c0 in range(p.shape[1])])
#%%
t0 = time.time()

RANDOM_STATE = 42

#%% CARGAR DATOS
data=pd.read_csv('../../datos/datos_generados/parsed/data_Cartera_parsed_CESAR.csv',index_col=0)
#%%
data.loc[(data['CLASE']>2),'CLASE'] = int(2)


data2016=data[data["ANUALIDAD"]==2016]
data2017=data[data["ANUALIDAD"]==2017]
data2018=data[data["ANUALIDAD"]==2018]

#data2016=data2016.drop(['COD_POLIZA','FEC_VIGENCIA','CAMPANIA', 'TIPO_DNI_TOM','IMP_O','REMOLQUE','EDAD_O','SIN_RC_ANIO_ANT',\
#                        'PAGO','DOMI', 'SUPL_PER_IRREG_ANT','ANUL_OTRA_POLIZA', 'RECL_ULT_ANIO','SINI_ULT_NO_CULP',\
#                        'MOROSOS','CLASIF_D4I_MOROSO','ESPECIAL_LA'], axis=1)
#
#data2017=data2017.drop(['COD_POLIZA','FEC_VIGENCIA','CAMPANIA', 'TIPO_DNI_TOM','IMP_O','REMOLQUE','EDAD_O','SIN_RC_ANIO_ANT',\
#                        'PAGO','DOMI', 'SUPL_PER_IRREG_ANT','ANUL_OTRA_POLIZA', 'RECL_ULT_ANIO','SINI_ULT_NO_CULP',\
#                        'MOROSOS','CLASIF_D4I_MOROSO','ESPECIAL_LA'], axis=1)
#
#data2018=data2018.drop(['COD_POLIZA','FEC_VIGENCIA','CAMPANIA', 'TIPO_DNI_TOM','IMP_O','REMOLQUE','EDAD_O','SIN_RC_ANIO_ANT',\
#                        'PAGO','DOMI', 'SUPL_PER_IRREG_ANT','ANUL_OTRA_POLIZA', 'RECL_ULT_ANIO','SINI_ULT_NO_CULP',\
#                        'MOROSOS','CLASIF_D4I_MOROSO','ESPECIAL_LA'], axis=1)

data2016=data2016.drop(['FEC_VIGENCIA'], axis=1)

data2017=data2017.drop(['FEC_VIGENCIA'], axis=1)

data2018=data2018.drop(['FEC_VIGENCIA'], axis=1)



#%%
dataTRAIN=pd.concat([data2016,data2017])
dataTEST=data2018.copy()

#%%
X_train=dataTRAIN.iloc[:,0:-1]
y_train=dataTRAIN.iloc[:,-1]

X_test=dataTEST.iloc[:,0:-1]
y_test=dataTEST.iloc[:,-1]

from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler()

X_train_normalization = sc.fit_transform(X_train)
X_test_normalization = sc.fit_transform(X_test)

#%%
clf_RandomForest=RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                   max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
                                   random_state=np.random.seed(0), class_weight=None)


clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 10), random_state=1)

#%%
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_train))
print('Resampled dataset shape %s' % Counter(y_res))

mi=mutual_info_classif(X_res, y_res, n_neighbors=3, copy=True, random_state=None)
mi2=mutual_info_classif(X_train, y_train, n_neighbors=3, copy=True, random_state=None)

print(mi2)
#%%

#sampler = ClusterCentroids(random_state=0)
#sampler= RandomUnderSampler(random_state=0)
#sampler=InstanceHardnessThreshold(random_state=42, estimator=LogisticRegression(solver='lbfgs', multi_class='auto'))
#sampler=NearMiss(version=1)

#classifier = make_pipeline(sampler, LinearSVC())
classifier=clf_RandomForest.fit(X_res, y_res)

#%%
#PARA TODAS LAS CLASES
#joblib.dump(classifier_RandomForest, "../../modelos/Algoritmo_03_2016_2017_2018_Random_Forest_CESAR/5CLASES/2016_2017_2018/classifier_RandomForest2" + ".pkl")
#joblib.dump(classifier_MLP, "../../modelos/Algoritmo_03_2016_2017_2018_Random_Forest_CESAR/5CLASES/2016_2017_2018/classifier_MLP2" + ".pkl")

#PARA SÓLO CLASES {0,1}
#joblib.dump(classifier_RandomForest, "../../modelos/Algoritmo_03_2016_2017_2018_Random_Forest_CESAR/5CLASES/2016_2017_2018/classifier_RandomForest3" + ".pkl")
#joblib.dump(classifier_MLP, "../../modelos/Algoritmo_03_2016_2017_2018_Random_Forest_CESAR/5CLASES/2016_2017_2018/classifier_MLP3" + ".pkl")


y_pred = classifier.predict(X_test)   
accuracyCLASE=accuracy_score(y_test, y_pred)*100
confusion_matrix_CLASES=confusion_matrix(y_test, y_pred)
p = classifier.predict_proba(X_test)
#%%
N_Clases=len(np.unique(y_test))
Vector_AUC=[]
for i in range(0,N_Clases):
    AUC= getAUC(p,dataTEST['CLASE'],i)
    Vector_AUC.append(AUC)
#%%
AUC_Avg=getAvgAUC(p,y_test)
#%%
t1 = time.time()
totalTime = t1-t0