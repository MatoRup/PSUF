import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from home_made_PCA import ploting
from home_made_PCA import tip_to_int
from sklearn.cluster import DBSCAN
from matplotlib import style
plt.style.use('ggplot')


def ploting_per(dimensions1,dimensions2,indexes_tip,tip,my_points,titles):

    figure, axis = plt.subplots(2, 2,figsize=(18, 15),constrained_layout=True)
    h=0
    for z in [0,1]:
        for j in [0,1]:
            dim1=dimensions1[h]
            dim2=dimensions2[h]
            axis[z, j].scatter(dim1,dim2,color ='red',s=2)
            axis[z, j].scatter(dim1[my_points],dim2[my_points], color='#13ADB7',label='Izbrani ekstremni primeri')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==1)]],dim2[indexes_tip[np.where(tip==1)]], color='blue',label='MAB')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==2)]],dim2[indexes_tip[np.where(tip==2)]], color='black',label='TRI')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==3)]],dim2[indexes_tip[np.where(tip==3)]], color='g',label='BIN')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==4)]],dim2[indexes_tip[np.where(tip==4)]], color='brown',label='HFR')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==5)]],dim2[indexes_tip[np.where(tip==5)]], color='yellow',label='HAE')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==6)]],dim2[indexes_tip[np.where(tip==6)]], color='pink',label='CMP')
            axis[z, j].scatter(dim1[indexes_tip[np.where(tip==7)]],dim2[indexes_tip[np.where(tip==7)]], color='purple',label='DIB')
            axis[z, j].legend()
            axis[z, j].title.set_text(titles[h])
            h=h+1


# load wavelengths
wav = np.loadtxt('spektri/val.dat', comments='#')
# load spectra in a lopos
spectra_array=[]
for spectrum in range(1,10000):
    flux = np.loadtxt('spektri/%s.dat' % spectrum, comments='#')
    spectra_array.append(flux)
# create an array
spectra_array = np.array(spectra_array, dtype='float64')

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


spectra_from_1=[2543, 262, 644, 4168]
test_per = True

if test_per:
    per = [5,30,60,250]
else :
    per = [150]

'''
8. Uporabite t-SNE, da naredite projekcijo podatkov v dve dimenziji. Preuˇcite kako prosti
parametri (ˇse posebej “perplexity”) te metode spremenijo projekcijo. Kje v tej vizualizaciji
leˇzijo spektri, ki ste jih naˇsli v 1. toˇcki? Enako kot za PCA, s pomoˇcjo uˇcnega seta
ugotovite, ali kakˇsna skupina zvezd v t-SNE projekciji predstavlja logiˇcno zakljuˇcen razred
zvezd.
'''
dimen1 = np.zeros((4,9999))
dimen2 = np.zeros((4,9999))
l=0
for a in per:

    tsne1 = TSNE(n_components=2,  init='random', perplexity=a)
    transformer = tsne1.fit_transform(spectra_array)
    dimen1[l] = transformer[:,0]
    dimen2[l] = transformer[:,1]
    plt.figure(figsize=(12,10))
    plt.scatter(dimen1[l],dimen2[l],s=2)
    #special spectras from ex1
    plt.scatter(dimen1[l][spectra_from_1],dimen2[l][spectra_from_1], color='r',s=2)
    plt.show()
    l=l+1

titles=['per=5','per=30','per=60','per=250']
ploting_per(dimen1,dimen2,indexes_tip,types,spectra_from_1,titles)
plt.savefig('per4.png')

iskanje = np.zeros((2,9999))
iskanje[0] = dimen1[2]
iskanje[1] = dimen2[2]

clustering = DBSCAN(eps=3,min_samples=10).fit(iskanje.T)

print(clustering.labels_)
for o in range(13,30):
    f, (ax1, ax2) = plt.subplots(2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    f.suptitle(str(o))
    ax1.scatter(dim1[np.where(clustering.labels_==1)], dim2[np.where(clustering.labels_==1)],s=2, color = 'blue')
    ax1.scatter(dim1[np.where(clustering.labels_==0)], dim2[np.where(clustering.labels_==0)],s=2, color = 'black')
    ax1.scatter(dim1[np.where(clustering.labels_==-1)], dim2[np.where(clustering.labels_==-1)],s=2, color = 'r')
    ax1.scatter(dim1[np.where(clustering.labels_==2)], dim2[np.where(clustering.labels_==2)],s=2, color = 'pink')
    ax1.scatter(dim1[np.where(clustering.labels_==3)], dim2[np.where(clustering.labels_==3)],s=2, color = 'yellow')
    ax1.scatter(dim1[np.where(clustering.labels_==4)], dim2[np.where(clustering.labels_==4)],s=2, color = 'yellow')
    a=ax1.scatter(dim1, dim2,s=2, c = clustering.labels_)
    ax1.scatter(dim1[np.where(clustering.labels_==o)], dim2[np.where(clustering.labels_==o)],s=2, color = 'r')
    bar3 = f.colorbar(a,ax=ax1)

6



f, (ax1, ax2) = plt.subplots(2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
ax1.scatter(dim1, dim2,s=2, color = 'blue')
ax1.scatter(dim1[np.where(clustering.labels_==4)], dim2[np.where(clustering.labels_== 4)],s=2, color = 'red')



grup1_m = np.mean(spectra_array[np.where(clustering.labels_ == 36),:],axis=1)[0,:]
grup2_m = np.mean(spectra_array[np.where(clustering.labels_ == 4),:],axis=1)[0,:]
grup3_m = np.mean(spectra_array[np.where(clustering.labels_ == -1),:],axis=1)[0,:]

grup1_std = np.std(spectra_array[np.where(clustering.labels_ == 36),:],axis=1)[0,:]
grup2_std = np.std(spectra_array[np.where(clustering.labels_ == 4),:],axis=1)[0,:]
grup3_std = np.std(spectra_array[np.where(clustering.labels_ == -1),:],axis=1)[0,:]


ax2.set_xlabel(r'Wavelength / $\mathrm{\AA}$')
ax2.set_ylabel(r'Normalized average flux of group')
ax2.fill_between(wav, grup2_m-grup2_std, grup2_m+grup2_std)
ax2.plot(wav, grup2_m, color='red')
plt.savefig('x_skpina.png')
