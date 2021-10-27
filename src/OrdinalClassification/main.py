import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import OrdinalClassif as OC

def getAUC(p, c, c0):
    '''
    p: probabilidad de la clase, p(C|x)
    c: array de clases reales (c,x)
    c0: la clase (positiva) para la que calculamos el AUC (int)
    '''
    c01 = np.array(c != c0, dtype=np.int)
    p0 = p[:, c0]

    (N0, N1) = np.unique(c01, return_counts=True)[1]
    delta = N0
    tau = 0

    sort = np.argsort(-p0)
    for i in range(len(sort)):
        if (c01[sort[i]] == 0):  # cuando es un cero
            delta -= 1
        else:  # cuando es un 1
            tau += delta
    AUC = 1.0 - (tau / (N0 * N1))
    return AUC


def getAvgAUC(p, c):
    return np.average([getAUC(p, c, c0) for c0 in range(p.shape[1])])

def test():
    data = pd.read_csv("../datos/datos_parsed/data_Cartera_parsed.csv", index_col=0)

    data=data[data["TIPO_COB6"]>0].drop(["TIPO_COB6"],axis=1)
        
    data.loc[(data['CLASE']>2),'CLASE'] = int(2)

    (n,d)= data.shape

    train0= data[data["ANUALIDAD"] == 2016].drop(['COD_POLIZA','FEC_VIGENCIA','ANUALIDAD'], axis=1)
    train1= data[data["ANUALIDAD"] == 2017].drop(['COD_POLIZA','FEC_VIGENCIA','ANUALIDAD'], axis=1)
    train=pd.concat([train0,train1])
    test= data[data["ANUALIDAD"] == 2018].drop(['COD_POLIZA','FEC_VIGENCIA','ANUALIDAD'], axis=1)

    Xtrain=train.iloc[:, 0:-1]
    Ytrain=train.iloc[:, -1]

    Xtest=test.iloc[:,0:-1]
    Ytest=test.iloc[:,-1]
    
    ordClassif = OC.ordinalClassif(RandomForestClassifier, 3)
    ordClassif.fit(Xtrain,Ytrain)

#    classif = RandomForestClassifier(400)
#    classif.fit(Xtrain,Ytrain)

    pOrd = ordClassif.predict_proba(Xtest)
#    p = classif.predict_proba(Xtest)

    AUCord= getAvgAUC(pOrd, Ytest)
    print(AUCord)
#    AUC= getAvgAUC(p, Ytest)
    print(str([getAUC(pOrd[:,0:3], Ytest, c0) for c0 in range(3)]))
#    print(str([getAUC(p[:,0:3], Ytest, c0) for c0 in range(3)]))

if __name__ == '__main__':
    test()