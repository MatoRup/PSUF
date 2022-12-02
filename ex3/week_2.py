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
								print(1)
						  
		print(self.hist)
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


def plot_predic2(y_real,y_predict,path_name,title):
	figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
	plt.scatter(y_real[:,0],y_predict[:,0],s=1)
	plt.plot([0,2e-11],[0,2e-11], c='r')
	plt.ylabel('$K1_{predvideni}$')
	plt.xlabel('$K1_{pravi}$')
	plt.title(title)
	plt.savefig(path_name+'1d.png')


	figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
	plt.hist2d(y_real[:,0],y_predict[:,0], bins=40)
	plt.ylabel('$K1_{predvideni}$')
	plt.xlabel('$K1_{pravi}$')
	plt.title(title)
	plt.colorbar()
	plt.savefig(path_name+'1dhist.png')

	figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
	plt.scatter(y_real[:,1],y_predict[:,1],s=1)
	plt.plot([0,2e-11],[0,2e-11], c='r')
	plt.ylabel('$K2_{predvideni}$')
	plt.xlabel('$K2_{pravi}$')
	plt.title(title)
	plt.savefig(path_name+'2d.png')


	figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
	plt.hist2d(y_real[:,1],y_predict[:,1], bins=40)
	plt.ylabel('$K2_{predvideni}$')
	plt.xlabel('$K2_{pravi}$')
	plt.title(title)
	plt.colorbar()
	plt.savefig(path_name+'2dhist.png')
	
def plot_predic(y_real,y_predict,path_name,title):
	figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
	plt.scatter(y_real,y_predict,s=1)
	plt.plot([0,2e-11],[0,2e-11], c='r')
	plt.ylabel('$K1_{predvideni}$')
	plt.xlabel('$K1_{pravi}$')
	plt.title(title)
	plt.savefig(path_name+'1d.png')







def TF_model(in_sha=400,optimizer='adam', neurons1=200,neurons2=100,learning_rate=0.001):
	model = Sequential ()
	model.add(Dense (neurons1, activation ='sigmoid', input_shape =(in_sha,)))
	model.add(Dense (neurons2, activation ='sigmoid'))
	model.add(Dense (2, activation ='relu'))
	if optimizer == 'adam':
		opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	elif optimizer == 'SGD':
		opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	model.compile(optimizer=opt, loss='mean_absolute_error', metrics =(['mean_squared_error']))
	return model
	
def printaj(D,model1):
	
	y_predict_predic = model1.predict(D)*MaxK
	y_predict_predic1 = y_predict_predic[:,0]
	y_predict_predic2 = y_predict_predic[:,1]
	print(y_predict_predic2)
	print('Cel set K1:')
	print(np.average(y_predict_predic1),np.std(y_predict_predic1))

	print('Cel set K2:')
	print(np.average(y_predict_predic2),np.std(y_predict_predic2))

	print('Samo 30 set K1:')
	print(np.average(y_predict_predic1[:30]),np.std(y_predict_predic1[:30]))

	print('Samo 30 set K2:')
	print(np.average(y_predict_predic2[:30]),np.std(y_predict_predic2[:30]))
	
####K-fold cross-validation####
def k_fold(x,y,res1):
	input = x[:predic_star,:]
	target = y[:predic_star,:]
	l=len(input[0])
	model_dic = dict(
	    models=[],
	    loss=[])

	for kfold, (train, test) in enumerate(KFold(n_splits=3, shuffle=True).split(input, target)):

	    # clear the session
	    tf.keras.backend.clear_session()

	    # calling the model and compile it
	    model = TF_model(optimizer=res1['optimizer'], neurons1=res1['neurons1'],neurons2=res1['neurons2'],learning_rate=res1['lr_values'],in_sha = l)

	    # run the model
	    model.fit(input[train], target[train], epochs=res1['epochs'], batch_size=res1['batch_size'], validation_data=(input[test], target[test]))
	    print(model.evaluate(input[test], target[test]))
	    evl = np.round_(model.evaluate(input[test], target[test])[1]*100, decimals = 10)
	    model_dic['loss'].append(evl)
	    model_dic['models'].append(model)
	return(model_dic['models'][np.argmin(model_dic['loss'])])



lam = ['500','505','700','900']
Data_path='/data/PSUF_naloge/3-naloga/DataK13_pp'
#Data_path='DataK13_pp'

DATA = np.load(Data_path+'/intensity'+lam[1]+'noise0.npy')
Kvalues = np.load(Data_path+'/Kvalues.npy')


predic_star = 90000
MaxK = np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK


####GridSearchCV###

batches = 30
epoch_values = 100
optimizers = 'adam'
neuron1_list = 250
neuron2_list = 100
lr_values = 0.001



param_grid1 = dict(
    epochs=epoch_values,
    batch_size=batches,
    optimizer=optimizers,
    neurons1=neuron1_list,
    neurons2=neuron2_list,
    lr_values = lr_values)


X_predict = DATA[predic_star:,:]


res1 = param_grid1


	    

final = k_fold(DATA,Kvalues_norm,res1 )

y_predict_predic = final.predict(X_predict)
y_predict_predic1 = y_predict_predic[:,0]
y_predict_predic2 = y_predict_predic[:,1]
plot_predic(Kvalues_norm[predic_star:,0]*MaxK,y_predict_predic1*MaxK,'Slike/brez suma 2k1',"  ")
plot_predic(Kvalues_norm[predic_star:,1]*MaxK,y_predict_predic2*MaxK,'Slike/brez suma 2k',"  ")



DATA0 = np.zeros((30,15600))

Data_path='/data/PSUF_naloge/3-naloga/ExpData'
#Data_path='ExpData'
for j in range(30):
    DATA0[j] = np.load(Data_path+'/exp_intensity_'+str(j)+'.npy')


#za 1,2t
DATA_final = np.zeros((30*24,400))
DATA1 = DATA0[:,:9600]
for j in range(24):
    DATA_final[j*30:(j+1)*30] = DATA1[:,np.arange(j,9600,24)]


print(DATA_final)

printaj(DATA_final,final)


l=100
DATA3 = fft(DATA,n=l, axis=- 1)



batches = 40
epoch_values = 90
optimizers = 'adam'
neuron1_list = 90
neuron2_list = 90
lr_values = 0.001


param_grid1 = dict(
    epochs=epoch_values,
    batch_size=batches,
    optimizer=optimizers,
    neurons1=neuron1_list,
    neurons2=neuron2_list,
    lr_values = lr_values)

final = k_fold(DATA3,Kvalues_norm,param_grid1 )


y_predict_predic = final.predict(DATA3[predic_star:,:])
y_predict_predic1 = y_predict_predic[:,0]
y_predict_predic2 = y_predict_predic[:,1]
plot_predic(Kvalues_norm[predic_star:,0]*MaxK,y_predict_predic1*MaxK,'Slike/brez suma FFT1',"  ")
plot_predic(Kvalues_norm[predic_star:,1]*MaxK,y_predict_predic2*MaxK,'Slike/brez suma FFT2',"  ")

S = fft(DATA_final,n=l, axis=- 1)

printaj(S,final)




Data_path='/data/PSUF_naloge/3-naloga/DataK13_pp'
DATA = np.load(Data_path+'/intensity'+lam[1]+'noise100.npy')


	    
final = k_fold(DATA,Kvalues_norm,res1 )


X_predict = DATA[predic_star:,:]

y_predict_predic = final.predict(X_predict)
y_predict_predic1 = y_predict_predic[:,0]
y_predict_predic2 = y_predict_predic[:,1]
plot_predic(Kvalues_norm[predic_star:,0]*MaxK,y_predict_predic1*MaxK,'Slike/brez s sumom 2k',"  ")
plot_predic(Kvalues_norm[predic_star:,1]*MaxK,y_predict_predic2*MaxK,'Slike/brez s sumom 2k2',"  ")


DATA3 = fft(DATA,n=l, axis=- 1)

final = k_fold(DATA3,Kvalues_norm,param_grid1 )

y_predict_predic = final.predict(DATA3[predic_star:,:])
y_predict_predic1 = y_predict_predic[:,0]
y_predict_predic2 = y_predict_predic[:,1]
plot_predic(Kvalues_norm[predic_star:,0]*MaxK,y_predict_predic1*MaxK,'Slike/brez s sumom FFT1',"  ")
plot_predic(Kvalues_norm[predic_star:,1]*MaxK,y_predict_predic2*MaxK,'Slike/brez s sumom FFT2',"  ")

S = fft(DATA_final,n=l, axis=- 1)

printaj(S,final)


