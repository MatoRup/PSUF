import numpy as np
import matplotlib.pyplot as plt
from home_made_PCA import PCA

# load wavelengths file
wav = np.loadtxt('spektri/val.dat', comments='#')
# load spectra in a loop
spectra_array=[]
for spectrum in range(1,5750):
    flux = np.loadtxt('spektri/%s.dat' % spectrum, comments='#')
    spectra_array.append(flux)
# create an array
spectra_array = np.array(spectra_array)

pca = PCA(2)
pca.fit(spectra_array)
transformer = pca.transform(spectra_array)

plt.scatter(transformer[:,0],transformer[:,1])
#2543 is the specter with max std
plt.scatter(transformer[2543,0],transformer[2543,1], color='r')
plt.show()
print(spectra_array.shape)
print(transformer.shape)
