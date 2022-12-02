import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from matplotlib import style
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scikeras.wrappers import KerasClassifier
from scipy.fftpack import fft

class Home_made_simple_GS:

    def __init__(self, param_grid= {'epochs': [20], 'batch_size': [30], 'optimizer': ['adam'], 'neurons1': [200], 'neurons2': [100], 'lr_values': [0.001]}):
        self.param_grid = param_grid
        self.best_comb = dict(
            epochs=[],
            batch_size=[],
            optimizer=[],
            neurons1=[],
            neurons2=[],
            loss = [],
            lr_values=[],
            val_loss = [])
        self.hist = dict(
            epochs=[],
            batch_size=[],
            optimizer=[],
            neurons1=[],
            neurons2=[],
            lr_values=[],
            loss = [],
            val_loss = [])

    def fit(self,X_train,y_train,X_test,y_test,in_sha=400):
        for opt in self.param_grid['optimizer']:
            for n1 in self.param_grid['neurons1']:
                for n2 in self.param_grid['neurons2']:
                    for ler in self.param_grid['lr_values']:
                        for btc in self.param_grid['batch_size']:
                            for epoc in self.param_grid['epochs']:
                                model = TF_model(in_sha=in_sha,optimizer=opt, neurons1=n1,neurons2=n2,learning_rate=ler)
                                history = model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs = epoc, batch_size =btc, verbose =0)
                                self.hist['epochs'].append(epoc)
                                self.hist['batch_size'].append(btc)
                                self.hist['lr_values'].append(ler)
                                self.hist['neurons2'].append(n2)
                                self.hist['neurons1'].append(n1)
                                self.hist['optimizer'].append(opt)
                                self.hist['loss'].append(history.history['loss'][-1])
                                self.hist['val_loss'].append(history.history['val_loss'][-1])

        index = np.argmin(self.hist['val_loss'])
        self.best_comb['epochs']=self.hist['epochs'][index]
        self.best_comb['batch_size']=self.hist['batch_size'][index]
        self.best_comb['neurons2']=self.hist['neurons2'][index]
        self.best_comb['neurons1']=self.hist['neurons1'][index]
        self.best_comb['optimizer']=self.hist['optimizer'][index]
        self.best_comb['loss']=self.hist['loss'][index]
        self.best_comb['val_loss']=self.hist['val_loss'][index]
        self.best_comb['lr_values']=self.hist['lr_values'][index]




def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.savefig('Slike/'+title+'.png')


def plot_predic(y_real,y_predict,path_name,title):
    figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
    plt.scatter(y_real,y_predict,s=1)
    plt.plot([0,2e-11],[0,2e-11], c='r')
    plt.ylabel('$K_{predvideni}$')
    plt.xlabel('$K_{pravi}$')
    plt.title(title)
    plt.ylim([0, 2.3e-11])
    plt.savefig(path_name+'.png')


    figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
    plt.hist2d(y_real,y_predict, bins=40)
    plt.ylabel('$K_{predvideni}$')
    plt.xlabel('$K_{pravi}$')
    plt.title(title)
    plt.savefig(path_name+'hist.png')




def TF_model(in_sha=400,optimizer='adam', neurons1=200,neurons2=100,learning_rate=0.001):
    model = Sequential ()
    model.add(Dense (neurons1, activation ='sigmoid', input_shape =(in_sha,)))
    model.add(Dense (neurons2, activation ='sigmoid'))
    model.add(Dense (1, activation ='relu'))
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics =(['mean_squared_error']))
    return model


lam = ['500','505','700','900']

Data_path='/DataK'
DATA = np.load(Data_path+'/intensity'+lam[0]+'noise0.npy')
Kvalues = np.load(Data_path+'/Kvalues.npy')
theta=  np.load(Data_path+'/theta0.npy')

figure = plt.subplots(figsize=(7, 4),constrained_layout=True)
plt.plot(np.linspace(0,1.20,400),DATA[3,:],label='C='+str(np.round_(Kvalues[3],decimals=13))+'N')
plt.plot(np.linspace(0,1.20,400),DATA[2,:],label='C='+str(np.round_(Kvalues[2],decimals=13))+'N')
plt.plot(np.linspace(0,1.20,400),DATA[1,:],label='C='+str(np.round_(Kvalues[1],decimals=13))+'N')
plt.plot(np.linspace(0,1.20,400),DATA[0,:],label='C='+str(np.round_(Kvalues[0],decimals=13))+'N')
plt.legend()
plt.ylabel('$I(t)$')
plt.xlabel('$t$')
plt.savefig('Slike/example1kfirstdata.png')


predic_star = 90000
MaxK = np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK

X_train, X_test, y_train, y_test = train_test_split( DATA[:predic_star,:], Kvalues_norm[:predic_star], test_size=0.20, random_state=42)

####GridSearchCV###


batches = [10, 30, 40,60,100]
epoch_values = [5,10, 20,50,60]
optimizers = ['adam', 'SGD']
neuron1_list = [100,250, 200,300]
neuron2_list = [30, 50,100,200]
lr_values = [0.001,0.01,0.1,1 ]


param_grid = dict(
    epochs=epoch_values,
    batch_size=batches,
    optimizer=optimizers,
    neurons1=neuron1_list,
    neurons2=neuron2_list,
    lr_values = lr_values)

GS = Home_made_simple_GS(param_grid)

GS.fit(X_train,y_train, X_test,y_test)

print(GS.hist['val_loss'])
print("Best: %s" % (GS.best_comb))

X_predict = DATA[predic_star:,:]
y_predict_rela = Kvalues_norm[predic_star:]
np.save("historgs1.txt", [GS.hist['epochs'],GS.hist['batch_size'],GS.hist['optimizer'],GS.hist['neurons1'],GS.hist['neurons2'],GS.hist['lr_values'],GS.hist['val_loss'],GS.hist['val_loss']])
res = GS.best_comb

for j in ['adam','SGD']:
    model = TF_model(optimizer=j, neurons1=res['neurons1'],neurons2=res['neurons2'],learning_rate=res['lr_values'])
    History = model.fit(X_train,y_train, epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(X_test,y_test))
    plot_train_history(History, 'Multi-Step Training and validation loss')
    y_predict_predic = model.predict(X_predict)

    plot_predic(y_predict_rela*MaxK,y_predict_predic[:,0]*MaxK,'Slike/BestGVresultK1'+j+'.png',j)


####K-fold cross-validation####
best_kfol=np.array([])
best_kfol_str=np.array([])
input = DATA[:predic_star,:]
target = Kvalues_norm[:predic_star]

for kfold, (train, test) in enumerate(KFold(n_splits=10, shuffle=True).split(input, target)):

    # clear the session
    tf.keras.backend.clear_session()

    # calling the model and compile it
    model = TF_model(optimizer=res['optimizer'], neurons1=res['neurons1'],neurons2=res['neurons2'],learning_rate=res['lr_values'])

    # run the model
    model.fit(input[train], target[train], epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(input[test], target[test]))
    print(model.evaluate(input[test], target[test]))
    evl = model.evaluate(input[test], target[test])[1]
    best_kfol =np.append(best_kfol, evl)
    s= 'weightK\wg_'+str(kfold)+str(evl)+'.h5'
    best_kfol_str =np.append(best_kfol_str, s)

final = tf.keras.models.load_model(best_kfol_str[np.argmax(best_kfol)])
y_predict_predic = final.predict(X_predict)
plot_predic(y_predict_rela*MaxK,y_predict_predic[:,0]*MaxK,'Slike/BestGVresultK1final.png',j)


#Šum#

DATA = np.load(Data_path+'/intensity'+lam[0]+'noise100.npy')
Kvalues = np.load(Data_path+'/Kvalues.npy')

figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
plt.plot(np.linspace(0,1.20,400),DATA[0,:])
plt.ylabel('$I(t)$')
plt.xlabel('$t$')
plt.savefig('Slike/example1knoise.png')

predic_star = 90000
MaxK = np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK



X_train, X_test, y_train, y_test = train_test_split(DATA[:predic_star,:], Kvalues_norm[:predic_star], test_size=0.20, random_state=42)

model = TF_model(optimizer=res['optimizer'], neurons1=res['neurons1'],neurons2=res['neurons2'],learning_rate=resres['lr_values'])
model.fit(X_train,y_train, epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(X_test,y_test))
plot_train_history(History, 'Multi-Step Training and validation loss noise')
y_predict_predic = model.predict(X_predict)
plot_predic(y_predict_rela*MaxK,y_predict_predic[:,0]*MaxK,'Slike/BestGVresultK1šum.png',res['optimizer'])

inal = tf.keras.models.load_model(best_kfol_str[np.argmax(best_kfol)])
y_predict_predic = final.predict(X_predict)
plot_predic(y_predict_rela*MaxK,y_predict_predic[:,0]*MaxK,'Slike/BestGVresultK1finalsum.png',res['optimizer'])

DATA = np.load(Data_path+'/intensity'+lam[0]+'noise0.npy')
Kvalues = np.load(Data_path+'/Kvalues.npy')
theta=  np.load(Data_path+'/theta0.npy')

figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
plt.plot(np.linspace(0,1.20,400),DATA[0,:])
plt.ylabel('$I(t)$')
plt.xlabel('$t$')
plt.savefig('Slike/example1kFFT.png')

l=100
DATA = fft(DATA,n=l, axis=- 1)


X_predict = DATA[predic_star:,:]
y_predict_rela = Kvalues_norm[predic_star:]

predic_star = 90000
MaxK = np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK

X_train, X_test, y_train, y_test = train_test_split(DATA[:predic_star,:], Kvalues_norm[:predic_star], test_size=0.20, random_state=42)

model = TF_model(in_sha = l,optimizer=res['optimizer'], neurons1=res['neurons1'],neurons2=res['neurons2'],learning_rate=0.001)
model.fit(X_train,y_train, epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(X_test,y_test))
plot_train_history(History, 'Multi-Step Training and validation loss for data with noise')
y_predict_predic = model.predict(X_predict)
plot_predic(y_predict_rela*MaxK,y_predict_predic[:,0]*MaxK,'Slike/BestGVresultK1šum.png',res['optimizer'])




####GridSearchCV###


batches = [10, 30, 40,60,100]
epoch_values = [5,10, 20,50,60]
optimizers = ['adam', 'SGD']
neuron1_list = [50,60, 70,80]
neuron2_list = [5, 10,20,30]
lr_values = [0.001,0.01,0.1,1 ]



param_grid = dict(
    epochs=epoch_values,
    batch_size=batches,
    optimizer=optimizers,
    neurons1=neuron1_list,
    neurons2=neuron1_list,
    lr_values = lr_values)

GS1 = Home_made_simple_GS(param_grid)

GS1.fit(in_sha = l,X_train,y_train, X_test,y_test)
np.save("historgs1.txt", [GS1.hist['epochs'],GS1.hist['batch_size'],GS1.hist['optimizer'],GS1.hist['neurons1'],GS1.hist['neurons2'],GS1.hist['lr_values'],GS1.hist['val_loss'],GS1.hist['val_loss']])
res = GS.best_comb
print(GS1.hist['val_loss'])
print("Best: %s" % (GS1.best_comb))

X_predict = DATA[predic_star:,:]
y_predict_rela = Kvalues_norm[predic_star:]

res = GS1.best_comb

for j in ['adam','SGD']:
    model = TF_model(in_sha = l, optimizer=j, neurons1=res['neurons1'],neurons2=res['neurons2'],learning_rate=res['lr_values'])
    History = model.fit(X_train,y_train, epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(X_test,y_test))
    plot_train_history(History, 'Multi-Step Training and validation loss FFT')
    y_predict_predic = model.predict(X_predict)

    plot_predic(y_predict_rela*MaxK,y_predict_predic[:,0]*MaxK,'Slike/BestGVresultK1FFT'+j+'.png',j)
