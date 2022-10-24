import numpy as np
import matplotlib.pyplot as plt
# load wavelengths file

'''
wav = np.loadtxt('spektri/val.dat', comments='#')
# load one spectrum (spectrum number 123, for example)
flux = np.loadtxt('spektri/123.dat', comments='#')
# plot spectrum
plt.plot(wav, flux, 'k-')
plt.xlabel(r'Wavelength / $\mathrm{\AA}$')
plt.ylabel(r'Normalized flux')
plt.show()
'''

# load wavelengths file
wav = np.loadtxt('spektri/val.dat', comments='#')
# load spectra in a loop
spectra_array=[]
for spectrum in range(1,5750):
    flux = np.loadtxt('spektri/%s.dat' % spectrum, comments='#')
    spectra_array.append(flux)
# create an array
spectra_array = np.array(spectra_array)
# plot spectrum
plt.plot(wav, spectra_array[1234], 'k-')
plt.xlabel(r'Wavelength / $\mathrm{\AA}$')
plt.ylabel(r'Normalized flux')
plt.show()

'''
1. Spoznajte se s podatkovnim setom. Najdite in nariˇsite nekaj spektrov, ki vizualno najbolj
odstopajo od povprečja.
'''

mean_valus_of_spectras = np.mean(spectra_array,axis=1)
std_valus_of_spectras = np.std(spectra_array,axis=1)


def plot_specter(wav, spectra_array, number):
    plt.plot(wav, spectra_array[number], 'k-')
    plt.title(r'Spectrum number '+str(number))
    plt.xlabel(r'Wavelength / $\mathrm{\AA}$')
    plt.ylabel(r'Normalized flux')
    plt.show()

plot_specter(wav, spectra_array,mean_valus_of_spectras.argmax())

plot_specter(wav, spectra_array,mean_valus_of_spectras.argmin())

plot_specter(wav, spectra_array, std_valus_of_spectras.argmin())

plot_specter(wav, spectra_array,std_valus_of_spectras.argmax())
print(std_valus_of_spectras.argmax())
