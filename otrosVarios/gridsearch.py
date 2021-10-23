from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, \
                            r2_score, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV


scoring = make_scorer(r2_score)


'''
defino el Regressor Tree
'''
regressorTree = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, 
                                  min_samples_split=2, min_samples_leaf=5, min_weight_fraction_leaf=0.0, 
                                  max_features=None, random_state=None, max_leaf_nodes=None, 
                                  min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

'''
Sobre el Regressor Tree ya definido, hago la busqueda de parametros. Aunque los parametros del RegressorTree ya están 
definidos arriba, esta busqueda los sustituye por el mejor que encuentre. 
'''

regressorTreeGS = GridSearchCV(regressorTree,
              param_grid={'min_samples_split': range(10,12), 'min_samples_leaf': range(4, 6),
                         'random_state': [0,42]},
              scoring=scoring, cv=10, refit=True, n_jobs=-1)

'''
Entreno
'''

modelTreeGS=regressorTreeGS.fit(X_train, y_train)

print('best accuracy',  modelTreeGS.best_score_ )
print('best parameters', modelTreeGS.best_params_ )
