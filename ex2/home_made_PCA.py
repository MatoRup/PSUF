import numpy as np
import matplotlib.pyplot as plt

# 2. Sprogramirajte metodo PCA in kodo priložite kot tekst na koncu v poročilo
# rešitev https://www.youtube.com/watch?v=52d7ha-GdV8

class PCA_home:

    def __init__(self, n_components):
        self.n_components = n_components
        self.compoents = None
        self.eigenvalus1 = None
        self.mean = None

    def fit(self,X):
        # calculating mean
        self.mean = np.mean(X,axis=0)
        X = X - self.mean
        # calculating covariance matrix
        # columns = features
        cov = np.cov(X.T)
        #eigenvector and eigenvalus
        eigenvalus, eigenvectors = np.linalg.eig(cov)
        #stor
        #[::-1] reverses the array
        indexes = np.argsort(eigenvalus)[::-1]
        eigenvalus = eigenvalus[indexes]
        eigenvector = eigenvectors[indexes]
        self.eigenvalus1 = eigenvalus
        # storing
        self.compoents = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X,self.compoents.T)

def ploting(dim1,dim2,indexes_tip,tip,indexes,Teff,logg,m_h,title):

    figure, axis = plt.subplots(2, 2,figsize=(18, 15),constrained_layout=True)
    figure.suptitle(title)
    axis[0, 0].scatter(dim1,dim2,color ='red',s=2)
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==1)]],dim2[indexes_tip[np.where(tip==1)]], color='blue',label='MAB')
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==2)]],dim2[indexes_tip[np.where(tip==2)]], color='black',label='TRI')
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==3)]],dim2[indexes_tip[np.where(tip==3)]], color='g',label='BIN')
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==4)]],dim2[indexes_tip[np.where(tip==4)]], color='brown',label='HFR')
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==5)]],dim2[indexes_tip[np.where(tip==5)]], color='yellow',label='HAE')
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==6)]],dim2[indexes_tip[np.where(tip==6)]], color='pink',label='CMP')
    axis[0, 0].scatter(dim1[indexes_tip[np.where(tip==7)]],dim2[indexes_tip[np.where(tip==7)]], color='purple',label='DIB')
    axis[0, 0].legend()

    axis[0, 1].scatter(dim1,dim2, Color='red',s=2)
    b = axis[0, 1].scatter(dim1[indexes],dim2[indexes], c=Teff)

    axis[1, 0].scatter(dim1,dim2, Color='red',s=2)
    c = axis[1, 0].scatter(dim1[indexes],dim2[indexes], c=logg)


    axis[1, 1].scatter(dim1,dim2, Color='red',s=2)
    a = axis[1, 1].scatter(dim1[indexes],dim2[indexes], c=m_h)

    bar1 = figure.colorbar(b,ax=axis[0, 1:])
    bar2 = figure.colorbar(a,ax=axis[1, 1:])
    bar3 = figure.colorbar(c,ax=axis[1, :1])

    bar1.set_label('$T_eff$')
    bar3.set_label('Logg')
    bar2.set_label('M/H')

def tip_to_int(types):
    for a in range(len(types)):
        if types[a] == 'MAB':
            types[a] = 1
        elif types[a] == 'TRI':
            types[a] = 2
        elif types[a] == 'BIN':
            types[a] = 3
        elif types[a] == 'HFR':
            types[a] = 4
        elif types[a] == 'HAE':
            types[a] = 5
        elif types[a] == 'CMP':
            types[a] = 6
        elif types[a] == 'DIB':
            types[a] = 7
    types = np.int_(types)
    return types
