import numpy as np

class ordinalClassif(object):
    '''
    Construct a classifier for ordinal supervised classification by means of the binary decomposition proposed in "A
    simple approach to ordinal classification [Hall et al. 2001]" using a probability classifier of scikitlearn as the
    base classifier
    '''

    def __init__(self, classif, r, params=None):
        '''
        :param classif: scikitlearn classifier
        :param r: the number of class values, i in [0,r-1]
        :param params: parameters for scikitlearn classifier
        :return: None
        '''

        self.r= r
        #falta meter los parametros
        self.model= [classif(400) for y in range(r-1)]

    def fit(self, Xtrain, Ytrain):
        '''
        Fit the binary classifiers that implement the ordinal classifier
        :param Xtrain: instances of the features, np.array((numinstances,numFeatures))
        :param Ytrain: instances of the class variable, no.array(numInstances)
        :return: None
        '''

        for y in range(self.r-1):
            #Binaryze the class variable: <=y is 0 and >y is 1
            Ybin= np.array(Ytrain > y, np.int)
            self.model[y].fit(Xtrain, Ybin)

    def predict_proba(self, Xtest):
        '''
        Return the class probability distribution for the unlabeled instances of Xtest
        :param Xtest: instances of the features, np.array((numInstances,numFeatures))
        :return: probability distribution of the class conditioned to each unlabeled instance, np.array((numInstances,numClasses))
        '''

        p= np.zeros((Xtest.shape[0],self.r))
        for y in range(self.r-2):
            #store p(<=y) for y=0,...,r-3
            p[:, y]= self.model[y].predict_proba(Xtest)[:,0]
        #store p(<=r-2) and p(=r-1)
        p[:, (self.r-2)]= self.model[self.r-2].predict_proba(Xtest)[:,0]

        #Monotone increasing: p(<=c) <= p(<=c+1), A.
        #for y in range(1,self.r-1):
        #Monotone increasing: p(<=c) <= p(<=c+1), B.
        for y in range((self.r-3), 0, -1):
            #p[:,y]= np.max(p[:,y-1:y+1],axis=1)#tipo A
            p[:,y]= np.min(p[:,y:y+2],axis=1)#tipo B
        p[:,self.r-1]= 1- p[:, (self.r-2)]

        #compute p(y)= p(<=y)-p(<=y-1)
        for y in range((self.r-2), 0, -1):
            p[:,y]= p[:,y]-p[:,(y-1)]

        return p