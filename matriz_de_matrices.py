import numpy as np

#%%
B = np.array([[2,5],[7,2]])
C = np.array([7,2,10])
D = np.array([0])
E = np.array([2])

#%%
A = np.array([[B,D], [D,E]])
#%%
print('B= ',B)
#%% 
# Imprime la posicion [0,0] de la matriz A, la cual es la matriz B, de esta ultima
# imprime la posicion [1,0], la cual es el numero 7
print(A[0][0][1][0])
