import numpy as np
import matplotlib.pyplot as plt
from home_made_PCA import PCA_home
from home_made_PCA import ploting
from home_made_PCA import tip_to_int
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import DBSCAN
import time
from matplotlib import style
plt.style.use('ggplot')


def ploting_3(dim1,dim2,Teff,logg,m_h,title):

    figure, axis = plt.subplots(1,3, figsize=(18, 9),constrained_layout=True)

    axis[0].scatter(dim1,dim2, Color='red',s=2)
    a = axis[0].scatter(dim1[indexes],dim2[indexes], c=Teff)

    axis[1].scatter(dim1,dim2, Color='red',s=2)
    b = axis[1].scatter(dim1[indexes],dim2[indexes], c=logg)


    axis[2].scatter(dim1,dim2, Color='red',s=2)
    c = axis[2].scatter(dim1[indexes],dim2[indexes], c=m_h)

    bar1 = figure.colorbar(a,ax=axis[0], location='bottom')
    bar2 = figure.colorbar(b,ax=axis[1], location='bottom')
    bar3 = figure.colorbar(c,ax=axis[2], location='bottom')

    bar1.set_label('$T_{eff}$')
    bar3.set_label('Logg')
    bar2.set_label('M/H')


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
indexes = np.int_(Fiz_podatki[:,0])-1
Teff = Fiz_podatki[:,1]
logg = Fiz_podatki[:,2]
m_h = Fiz_podatki[:,3]

#Data from learning set for types of stars
Star_types = np.loadtxt('ucni_tip.txt', comments='#', dtype='str')
indexes_tip = np.int_(Star_types[:,0])-1
types = Star_types[:,1]
types = tip_to_int(types)

#home made
PCA_home = PCA_home(7)
pca = PCA(7)

time_1 = time.time()
PCA_home.fit(spectra_array)
time_2 = time.time()

print("Home made fit function working time: "+ str(time_2 - time_1))

time_1 = time.time()
pca.fit(spectra_array)
time_2 = time.time()

print("Sklearn fit function  working time: "+ str(time_2 - time_1))

transformer1 = PCA_home.transform(spectra_array)
transformer = pca.transform(spectra_array)

spectra_from_1=[2543, 262, 644, 4168]
dim1 = transformer[:,0]
dim2 = transformer[:,1]

fig = plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(dim1,dim2,s=2, color='blue',label='Vsi spektri')
#special spectras from ex1
plt.scatter(dim1[spectra_from_1],dim2[spectra_from_1], color='r',s=20,label='Izbrani ekstremni primeri')
plt.title('Scikit-learn alghoritem')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(transformer1[:,0],transformer1[:,1],s=2, color='blue',label='Vsi spektri')
#special spectras from ex1
plt.scatter(transformer1[:,0][spectra_from_1],transformer1[:,1][spectra_from_1], color='r',s=20,label='Izbrani ekstremni primeri')
plt.title('Moj alghoritem')
plt.legend()
plt.tight_layout()
plt.savefig('PCA.jpg')
plt.show()

'''
Najmanj koliko dimenzij potrebujete, da dobro opišete dani set spektrov? Dober opis
pomeni da zajamemo bistveno večino variance podatkov. V tem smislu raziščite katera
fizikalna količina je odgovorna za najveˇc variance in katera se najprej ”porazgubi“ v projekcijskem prostoru ko zmanjˇsujemo ˇstevilo dimenzij.
'''

eigenvalus = PCA_home.eigenvalus1

fig = plt.figure(figsize=(10, 6))
plt.scatter(range(0,len(eigenvalus[:])),eigenvalus[:])
plt.ylabel(r'Lastna vrednos $\lambda_j$')
plt.xlabel('j')
plt.yscale('log')
plt.savefig('PCA_vsi.jpg')
plt.show()

sestev = np.sum(eigenvalus)
a=0
while(True):
    a = a+1
    procent = np.sum(eigenvalus[:a])/np.sum(eigenvalus)
    if procent >0.9:
        print(a)
        print(procent)
        print(np.sum(eigenvalus[:a-1])/np.sum(eigenvalus))
        break

'''
m = np.mean(spectra_array,axis=0)
m = spectra_array - m
cov = np.cov(m.T)

f = plt.figure(figsize=(19, 15))
plt.imshow(np.log(np.abs(cov)),aspect='auto',extent=[np.min(wav), np.max(wav), 1, len(spectra_array[:,0])])
plt.xlabel(r'Wavelength / $\mathrm{\AA}$')
plt.show()
'''



'''
S pomoˇcjo uˇcnega seta ugotovite, ali kakˇsna skupina zvezd v PCA projekciji predstavlja
logiˇcno zakljuˇcen razred zvezd (ki mu pripada neka unikatna fizikalna znaˇcilnost ali veˇc
le-teh). Ali v delu spektrov, ki ne paˇse v nobeno izolirano skupino, oziroma tvori najveˇcjo
skupino, opazite kakˇsne trende?
'''


more_dim = False

for  i in range(7):
    for  a in range(i,7):
        dim1 = transformer[:,i]
        dim2 = transformer[:,a]
        title = 'Dim:'+str(i)+','+ str(a)
        if i<2 and a<2 and i!=a:
            ploting_3(dim1,dim2,indexes,Teff,logg,m_h)
        elif  more_dim and  i!=a :
            ploting_3(dim1,dim2,indexes,Teff,logg,m_h)
plt.savefig('trije.png')

'''
7. Uporabnost PCA primerjajte z variacijo metode po imenu kernel PCA, ki jo dobite v npr.
python knjiˇznicah.
'''

#If I use laplacian there is a difference otherwise no
Kpca = KernelPCA(7,kernel="laplacian")
transformer = Kpca.fit_transform(spectra_array)

for  i in range(6):
    for  a in range(i,6):
        dim1 = transformer[:,i]
        dim2 = transformer[:,a]
        title = 'Dim:'+str(i)+','+ str(a)
        if i<2 and a<2 and i!=a:
            ploting(dim1,dim2,indexes_tip,types,indexes,Teff,logg,m_h,title)
            plt.savefig('PCA_laplacian.jpg')
        elif  more_dim and  i!=a :

            ploting(dim1,dim2,indexes_tip,types,indexes,Teff,logg,m_h,title)
            ploting(dim1,dim2,indexes_tip,types,indexes,Teff,logg,m_h,title)



transformer = pca.transform(spectra_array)
dim1 = transformer[:,0]
dim2 = transformer[:,4]
clustering = DBSCAN(eps=0.3, min_samples=5).fit(transformer[:,[0,4]])

f, (ax1, ax2) = plt.subplots(2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
ax1.scatter(dim1[np.where(clustering.labels_==1)], dim2[np.where(clustering.labels_==1)],s=2, color = 'blue')
ax1.scatter(dim1[np.where(clustering.labels_==0)], dim2[np.where(clustering.labels_==0)],s=2, color = 'blue')
ax1.scatter(dim1[np.where(clustering.labels_==-1)], dim2[np.where(clustering.labels_==-1)],s=2, color = 'r')
ax1.scatter(dim1[np.where(clustering.labels_==2)], dim2[np.where(clustering.labels_==2)],s=2, color = 'blue')
ax1.scatter(dim1[indexes_tip[np.where(types==1)]],dim2[indexes_tip[np.where(types==1)]], color='r',label='MAB')
ax1.scatter(dim1[indexes_tip[np.where(types==2)]],dim2[indexes_tip[np.where(types==2)]], color='black',label='TRI')
ax1.scatter(dim1[indexes_tip[np.where(types==3)]],dim2[indexes_tip[np.where(types==3)]], color='g',label='BIN')
ax1.scatter(dim1[indexes_tip[np.where(types==4)]],dim2[indexes_tip[np.where(types==4)]], color='brown',label='HFR')
ax1.scatter(dim1[indexes_tip[np.where(types==5)]],dim2[indexes_tip[np.where(types==5)]], color='yellow',label='HAE')
ax1.scatter(dim1[indexes_tip[np.where(types==6)]],dim2[indexes_tip[np.where(types==6)]], color='pink',label='CMP')
ax1.scatter(dim1[indexes_tip[np.where(types==7)]],dim2[indexes_tip[np.where(types==7)]], color='purple',label='DIB')
ax1.legend()




grup1_m = np.mean(spectra_array[np.where(clustering.labels_ == 2),:],axis=1)[0,:]
grup2_m = np.mean(spectra_array[np.where(clustering.labels_ == -1),:],axis=1)[0,:]
grup3_m = np.mean(spectra_array[np.where(clustering.labels_ == -1),:],axis=1)[0,:]

grup1_std = np.std(spectra_array[np.where(clustering.labels_ == 2),:],axis=1)[0,:]
grup2_std = np.std(spectra_array[np.where(clustering.labels_ == -1),:],axis=1)[0,:]
grup3_std = np.std(spectra_array[np.where(clustering.labels_ == -1),:],axis=1)[0,:]


ax2.set_xlabel(r'Wavelength / $\mathrm{\AA}$')
ax2.set_ylabel(r'Normalized average flux of group')
ax2.fill_between(wav, grup2_m-grup2_std, grup2_m+grup2_std)
ax2.plot(wav, grup2_m, color='black')
plt.savefig('PCA_skupine2.jpg')
plt.show()


plt.figure(figsize=(12,10))
plt.plot(wav, grup1_m, 'k-')
plt.xlabel(r'Wavelength / $\mathrm{\AA}$')
plt.ylabel(r'Normalized average flux of group')
plt.fill_between(wav, grup1_m-grup1_std, grup1_m+grup1_std)
plt.show()
