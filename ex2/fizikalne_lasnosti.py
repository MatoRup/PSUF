import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from home_made_PCA import PCA_home
plt.style.use('ggplot')

#load wavelengths
wav = np.loadtxt('spektri/val.dat', comments='#')
#load spectra in a lopos
spectra_array=[]
for spectrum in range(1,10000):
    flux = np.loadtxt('spektri/%s.dat' % spectrum, comments='#')
    spectra_array.append(flux)
#create an array
spectra_array = np.array(spectra_array)

#Data from learning set
Fiz_podatki = np.loadtxt('ucni_set.txt', comments='#')
indexes = np.int_(Fiz_podatki[:,0])
Teff = Fiz_podatki[:,1]
logg = Fiz_podatki[:,2]
m_h = Fiz_podatki[:,3]

PCA_home = PCA_home(7)
PCA_home.fit(spectra_array[indexes])
transformer1 = PCA_home.transform(spectra_array[indexes])


eigenvalus = np.real(PCA_home.eigenvalus1)**2

eigenvalus = np.sqrt(eigenvalus)
fig = plt.figure(figsize=(10, 6))
plt.scatter(range(0,len(eigenvalus)),eigenvalus)
plt.ylabel(r'Lastna vrednos $\lambda_j$')
plt.xlabel('j')
plt.yscale('log')
plt.savefig('PCA_fiz.jpg')
plt.show()
