import numpy as np

# 2. Sprogramirajte metodo PCA in kodo priložite kot tekst na koncu v poročilo
# rešitev https://www.youtube.com/watch?v=52d7ha-GdV8

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.compoents = None
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
        # storing
        self.compoents = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X,self.compoents.T)
