import numpy as np
import matplotlib.pyplot as plt
from home_made_PCA import PCA_home
from sklearn.decomposition import PCA
import time

# load wavelengths file
wav = np.loadtxt('spektri/val.dat', comments='#')
# load spectra in a loop
spectra_array=[]
for spectrum in range(1,5750):
    flux = np.loadtxt('spektri/%s.dat' % spectrum, comments='#')
    spectra_array.append(flux)
# create an array
spectra_array = np.array(spectra_array)

#home made
PCA_home = PCA_home(3)
pca = PCA(3)

time_1 = time.time()
PCA_home.fit(spectra_array)
time_2 = time.time()

print("Home made fit function working time: "+ str(time_2 - time_1))

time_1 = time.time()
pca.fit(spectra_array)
time_2 = time.time()

print("Sklearn fit function  working time: "+ str(time_2 - time_1))

transformer = PCA_home.transform(spectra_array)
transformer = pca.transform(spectra_array)

spectra_from_1=[2543,2579,2211,262]
dim1 = transformer[:,0]
dim2 = transformer[:,1]

plt.scatter(dim1,dim2)
#specal spectras from ex1
plt.scatter(dim1[spectra_from_1],dim1[spectra_from_1], color='r')
plt.show()

'''
Najmanj koliko dimenzij potrebujete, da dobro opišete dani set spektrov? Dober opis
pomeni da zajamemo bistveno večino variance podatkov. V tem smislu raziščite katera
fizikalna količina je odgovorna za najveˇc variance in katera se najprej ”porazgubi“ v projekcijskem prostoru ko zmanjˇsujemo ˇstevilo dimenzij.
'''

eigenvalus = PCA_home.eigenvalus1


plt.scatter(range(0,len(eigenvalus)),eigenvalus)
plt.yscale('log')
plt.ylabel(r'eigenvalu')
plt.show()

'''
m = np.mean(spectra_array,axis=0)
m = spectra_array - m
cov = np.cov(m.T)

f = plt.figure(figsize=(19, 15))
plt.imshow(np.log(np.abs(cov)),aspect='auto',extent=[np.min(wav), np.max(wav), 1, len(spectra_array[:,0])])
plt.xlabel(r'Wavelength / $\mathrm{\AA}$')
plt.show()
'''

Fiz_podatki = np.loadtxt('ucni_set.txt', comments='#')

indexes = np.int_(Fiz_podatki[:,0])
Teff = Fiz_podatki[:,1]
logg = Fiz_podatki[:,2]
m_h = Fiz_podatki[:,3]

figure, axis = plt.subplots(2, 2,figsize=(18, 15),constrained_layout=True)

axis[0, 0].scatter(dim1,dim2)
axis[0, 0].scatter(dim1[spectra_from_1],dim2[spectra_from_1], color='r')

axis[0, 1].scatter(dim1,dim2, Color='r')
b = axis[0, 1].scatter(dim1[indexes],dim2[indexes], c=Teff)

axis[1, 0].scatter(dim1,dim2, Color='r')
c = axis[1, 0].scatter(dim1[indexes],dim2[indexes], c=logg)


axis[1, 1].scatter(dim1,dim2, Color='r')
a = axis[1, 1].scatter(dim1[indexes],dim2[indexes], c=m_h)

bar1 = figure.colorbar(b,ax=axis[0, 1:])
bar2 = figure.colorbar(a,ax=axis[1, 1:])
bar3 = figure.colorbar(c,ax=axis[1, :1])

bar1.set_label('T_eff')
bar2.set_label('Logg')
bar2.set_label('M/H')

plt.show()
