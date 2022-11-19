import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from home_made_PCA import ploting
from sklearn.cluster import DBSCAN
from home_made_PCA import tip_to_int
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA

def int_to_tip(types):
    s =np.array([])
    for a in range(len(types)):
        if types[a] == 1:
            s=np.append(s,'MAB')
        elif types[a] == 2:
            s=np.append(s,'TRI')
        elif types[a] == 3:
            s=np.append(s,'BIN')
        elif types[a] == 4:
            s=np.append(s,'HFR')
        elif types[a] == 5:
            s=np.append(s,'HAE')
        elif types[a] == 6:
            s=np.append(s,'CMP')
        elif types[a] == 7:
            s=np.append(s,'DIB')
    return s


# load wavelengths
wav = np.loadtxt('spektri/val.dat', comments='#')
# load spectra in a lopos
spectra_array=[]
for spectrum in range(1,9999):
    flux = np.loadtxt('spektri/%s.dat' % spectrum, comments='#')
    spectra_array.append(flux)
# create an array
spectra_array = np.array(spectra_array, dtype='float64')


#Data from learning set for types of stars
Star_types = np.loadtxt('ucni_tip.txt', comments='#', dtype='str')
indexes = np.int_(Star_types[:,0])-1
types = Star_types[:,1]
types = tip_to_int(types)


Kpca = KernelPCA(110,kernel="laplacian")
transformer1 = Kpca.fit_transform(spectra_array)

tsne1 = TSNE(n_components=3, perplexity=60)
transformer = tsne1.fit_transform(transformer1)
model = SVC()
model.fit(transformer[indexes,:],types)
y = model.predict(transformer)



figure = plt.subplots(figsize=(15, 10),constrained_layout=True)

dim1=transformer[:,1]
dim2=transformer[:,2]

plt.scatter(dim1[indexes[np.where(types==1)]],dim2[indexes[np.where(types==1)]], color='blue',label='MAB',s=100)
plt.scatter(dim1[indexes[np.where(types==2)]],dim2[indexes[np.where(types==2)]], color='black',label='TRI',s=100)
plt.scatter(dim1[indexes[np.where(types==3)]],dim2[indexes[np.where(types==3)]], color='g',label='BIN',s=100)
plt.scatter(dim1[indexes[np.where(types==4)]],dim2[indexes[np.where(types==4)]], color='brown',label='HFR',s=100)
plt.scatter(dim1[indexes[np.where(types==5)]],dim2[indexes[np.where(types==5)]], color='yellow',label='HAE',s=100)
plt.scatter(dim1[indexes[np.where(types==6)]],dim2[indexes[np.where(types==6)]], color='pink',label='CMP',s=100)
plt.scatter(dim1[indexes[np.where(types==7)]],dim2[indexes[np.where(types==7)]], color='purple',label='DIB',s=100)

plt.scatter(dim1[np.where(y==1)],dim2[np.where(y==1)], color='blue',s=4)
plt.scatter(dim1[np.where(y==2)],dim2[np.where(y==2)], color='black',s=4)
plt.scatter(dim1[np.where(y==3)],dim2[np.where(y==3)], color='g',s=4)
plt.scatter(dim1[np.where(y==4)],dim2[np.where(y==4)], color='brown',s=4)
plt.scatter(dim1[np.where(y==5)],dim2[np.where(y==5)], color='yellow',s=4)
plt.scatter(dim1[np.where(y==6)],dim2[np.where(y==6)], color='pink',s=4)
plt.scatter(dim1[np.where(y==7)],dim2[np.where(y==7)], color='purple',s=4)

plt.legend()
plt.savefig('klasifikacija2.png')
plt.show()


grup1_m = np.zeros((7,2084))
grup1_std = np.zeros((7,2084))

for z in range(0,7):
    print(np.mean(spectra_array[np.where(y == z+1)],axis=1))
    grup1_m[z] = np.mean(spectra_array[np.where(y == z+1),:],axis=1)[0,:]
    grup1_std[z] = np.std(spectra_array[np.where(y == z+1),:],axis=1)[0,:]

fig, axs = plt.subplots(7,figsize=(18, 15), sharex=True)
axs[0].plot(wav, grup1_m[0], color='r')
axs[0].fill_between(wav, grup1_m[0]-grup1_std[0], grup1_m[0]+grup1_std[0])
axs[0].set_title("MAB-zvezde z molekulskimi absorpcijskimi črtami")
axs[1].plot(wav, grup1_m[1], color='black')
axs[1].fill_between(wav, grup1_m[1]-grup1_std[1], grup1_m[1]+grup1_std[1])
axs[1].set_title('TRI– trojne zvezde')
axs[2].plot(wav, grup1_m[2], color='g')
axs[2].fill_between(wav, grup1_m[2]-grup1_std[2], grup1_m[2]+grup1_std[2])
axs[2].set_title('BIN– dvojne zvezde')
axs[3].plot(wav, grup1_m[3], color='brown')
axs[3].fill_between(wav, grup1_m[3]-grup1_std[3], grup1_m[3]+grup1_std[3])
axs[3].set_title('HFR– vroče, hitro vrteče se zvezde')
axs[4].plot(wav, grup1_m[4], color='yellow')
axs[4].fill_between(wav, grup1_m[4]-grup1_std[4], grup1_m[4]+grup1_std[4])
axs[4].set_title('HAE– zvezde z emisijo $H_{alpha}$')
axs[5].plot(wav, grup1_m[5], color='pink')
axs[5].fill_between(wav, grup1_m[5]-grup1_std[5], grup1_m[5]+grup1_std[5])
axs[5].set_title('CMP– hladne zvezde z malo kovinami')
axs[6].plot(wav, grup1_m[6], color='purple')
axs[6].fill_between(wav, grup1_m[6]-grup1_std[6], grup1_m[6]+grup1_std[6])
axs[6].set_title('DIB – vroče zvezde z močnejšimi medzvezdnimi absorpcijami ')
fig.text(0.5, 0.04, r'Wavelength / $\mathrm{\AA}$', ha='center', va='center')
fig.text(0.06, 0.5, r'Normalized flux', ha='center', va='center', rotation='vertical')
plt.savefig('final_specters.png')
plt.show()


tipi = int_to_tip(y)
print(tipi)

np.savetxt('final.txt', tipi.T, fmt='%s', newline='\n')

plt.figure(figsize=(12,10))
plt.scatter(transformer[:, 1], transformer[:, 2],s=2, c = y)
plt.show()

clustering = DBSCAN(eps=0.060,min_samples=10).fit(transformer1)

np.max(clustering.labels_)

grup1_m = np.zeros((10,2084))
grup1_std = np.zeros((10,2084))

for z in range(0,10):
    grup1_m[z] = np.mean(spectra_array[np.where(clustering.labels_ == z-1),:],axis=1)[0,:]
    grup1_std[z] = np.std(spectra_array[np.where(clustering.labels_ == z-1),:],axis=1)[0,:]

fig, axs = plt.subplots(10,figsize=(18, 15), sharex=True)
for l in range(0,10):
    axs[l].plot(wav, grup1_m[l], color='r')
    axs[l].fill_between(wav, grup1_m[l]-grup1_std[l], grup1_m[l]+grup1_std[l])
    axs[l].set_title(str(l+1))
fig.text(0.5, 0.04, r'Wavelength / $\mathrm{\AA}$', ha='center', va='center')
fig.text(0.06, 0.5, r'Normalized flux', ha='center', va='center', rotation='vertical')
plt.savefig('final_specters2.png')
plt.show()


out =np.column_stack((range(1,9999),tipi))
out =np.column_stack((out,clustering.labels_))
print(out)

np.savetxt('final.txt', out, fmt='%s', newline='\n')
