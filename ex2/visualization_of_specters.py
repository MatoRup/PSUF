import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('ggplot')

# load wavelengths file
wav = np.loadtxt('spektri/val.dat', comments='#')
# load spectra in a loop
spectra_array=[]
for spectrum in range(1,10000):
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

fig, axs = plt.subplots(7,figsize=(18, 15), sharex=True, sharey=True)
axs[0].plot(wav, spectra_array[46], color='r')
axs[0].set_title("MAB-zvezde z molekulskimi absorpcijskimi črtami")
axs[1].plot(wav, spectra_array[517], color='black')
axs[1].set_title('TRI– trojne zvezde')
axs[2].plot(wav, spectra_array[2122], color='g')
axs[2].set_title('BIN– dvojne zvezde')
axs[3].plot(wav, spectra_array[5043], color='brown')
axs[3].set_title('HFR– vroče, hitro vrteče se zvezde')
axs[4].plot(wav, spectra_array[1362], color='yellow')
axs[4].set_title('HAE– zvezde z emisijo $H_{alpha}$')
axs[5].plot(wav, spectra_array[3222], color='pink')
axs[5].set_title('CMP– hladne zvezde z malo kovinami')
axs[6].plot(wav, spectra_array[3384], color='purple')
axs[6].set_title('DIB – vroče zvezde z močnejšimi medzvezdnimi absorpcijami ')
fig.text(0.5, 0.04, r'Wavelength / $\mathrm{\AA}$', ha='center', va='center')
fig.text(0.06, 0.5, r'Normalized flux', ha='center', va='center', rotation='vertical')
plt.savefig('tipi_spektrov.jpg')
plt.show()


std_valus_of_spectras = np.std(spectra_array,axis=1)[2544:]

maxvalues = spectra_array.max(axis=1)
minvalues = spectra_array.min(axis=1)


fig, axs = plt.subplots(4,figsize=(18, 15), sharex=True)
axs[0].plot(wav, spectra_array[maxvalues.argmax()], color='r')
axs[0].set_title("Spekter z največjo pozitivno izmerjeno amplitudo")
axs[1].plot(wav, spectra_array[minvalues.argmin()], color='black')
axs[1].set_title('Spekter z največjo negativno izmerjeno amplitudo')
axs[2].plot(wav, spectra_array[std_valus_of_spectras.argmax()], color='g')
axs[2].set_title('Spekter z drugim največjim standardnim odklonom amplitude')
axs[3].plot(wav, spectra_array[std_valus_of_spectras.argmin()], color='brown')
axs[3].set_title('Spekter z najmanjšim standardnim odklonom amplitude')
fig.text(0.5, 0.04, r'Wavelength / $\mathrm{\AA}$', ha='center', va='center')
fig.text(0.06, 0.5, r'Normalized flux', ha='center', va='center', rotation='vertical')
plt.savefig('ekstremi.jpg')
plt.show()


print(maxvalues.argmax(),minvalues.argmax(),std_valus_of_spectras.argmax(),std_valus_of_spectras.argmin())
