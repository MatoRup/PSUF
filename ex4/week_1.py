from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow import keras
from tensorflow.keras import layers, initializers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt

def make_model(input_dim=3):

    # inicializiramo mrezo
    model = tf.keras.models.Sequential()
    # vhodni sloj 3 nevronov in plast 81
    model.add(tf.keras.layers.Dense(81, input_dim=3, activation='tanh'))
    # preostali skriti sloji z 81 nevroni
    model.add(tf.keras.layers.Dense(81, activation='tanh'))
    model.add(tf.keras.layers.Dense(81, activation='tanh'))
    model.add(tf.keras.layers.Dense(81, activation='tanh'))
    # izhodni sloj 3 nevronov
    model.add(tf.keras.layers.Dense(input_dim, activation='sigmoid'))
    # criterion for early stopping of netwo
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae', 'mean_squared_error'])
    return model

def model_RNN(nevr,opti,time_steps_predictor,num):
    model_rnn = tf.keras.models.Sequential()
    model_rnn.add(SimpleRNN(16, activation='tanh',input_shape=(time_steps_predictor,3)))
    for z in range(num):
    	model_rnn.add(Dense(nevr, activation='tanh'))
    model_rnn.add(Dense(3))
    model_rnn.compile(loss='mean_squared_error', optimizer=opti)
    return model_rnn

def model_LSTM(nevr,opti,time_steps_predictor,num):
    model_LSTM = tf.keras.models.Sequential()
    model_LSTM.add(LSTM(16, activation='tanh',input_shape=(time_steps_predictor,3)))
    for z in range(num):
    	model_rnn.add(Dense(nevr, activation='tanh'))
    model_LSTM.add(Dense(3))
    model_LSTM.compile(loss='mean_squared_error', optimizer=opti)
    return model_LSTM

def model_GRU(nevr,opti,time_steps_predictor,num):
    model_GRU = tf.keras.models.Sequential()
    model_GRU.add(GRU(16, activation='tanh',input_shape=(time_steps_predictor,3)))
    for z in range(num):
    	model_rnn.add(Dense(nevr, activation='tanh'))
    model_GRU.add(Dense(3))
    model_GRU.compile(loss='mean_squared_error', optimizer=opti)
    return model_GRU



def split_sequences(sequences, n_steps):
    X, y = list(), list()

    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plot_train_history(history, title):
    plt.style.use('ggplot')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(figsize=(10,6))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('Slike/'+title+'.png')


def plot_series(Y_real,Y_predicted,save,nn=['nevronska mreža']):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3,figsize=(10,6), sharex=True, sharey=True)
    for d in range(3):
        t = len(Y_real[:,d])
        axs[d].plot(np.linspace(0,t*0.01,t),Y_real[:,d],label='integrator')
        s = ['solid','dashed','dashdot','--','-.']
        if len(nn)==1:
            axs[d].plot(np.linspace(0,t*0.01,t),Y_predicted[:,d],label=nn[0],linestyle=s[0])
        else:
            for u in range(len(nn)):
                axs[d].plot(np.linspace(0,t*0.01,t),Y_predicted[u][:,d],label=nn[u],linestyle=s[u])
    axs[2].legend()
    fig.text(0.5, 0.04, 'time', ha='center')
    axs[0].set_ylabel('X')
    axs[1].set_ylabel('Y')
    axs[2].set_ylabel('Z')
    plt.savefig('Slike/'+save+'.png')

def predict(model,num_pedict,num_starih,n=0):
    Y_predicted = np.zeros((num_pedict,3))
    if np.isscalar(num_starih[0]):
        l=1
    else:
        l = len(num_starih[:,0])-n
    Y_predicted[:l] = num_starih
    for z in range(num_pedict-l+n):
        Y_predicted[z+l+n] = model.predict(Y_predicted[z:z+l].reshape((1,l,3)))

    return Y_predicted


def k_fold(fun, input,target,opti,nevroni,epochs):
    model_dic = dict(models=[],loss=[],hist=[])
    for kfold, (train, test) in enumerate(KFold(n_splits=3, shuffle=True).split(input, target)):
        # clear the session
        tf.keras.backend.clear_session()
        # calling the model and compile it
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, verbose=0, mode='min')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5, min_lr=0.0001)
        model = fun(nevroni,opti,time_steps_predictor)
        history = model.fit(input[train], target[train], epochs=epochs , validation_data=(input[test], target[test]),callbacks=[early_stop,reduce_lr])
        evl = model.evaluate(input[test], target[test])
        model_dic['loss'].append(evl)
        model_dic['models'].append(model)
        model_dic['hist'].append(history)

    naj = np.argmin(model_dic['loss'])
    return model_dic['models'][naj],model_dic['hist'][naj],np.min(model_dic['loss'])

def GS(fun,input,target,params):
    podatki = dict(
        models=[],
        hist=[],
        los=[],
        opti= [],
        nerv = [],
        epochs = [])

    for opti in params['opti']:
        for ner in params['nerv']:
            for epo in params['epochs']:
                model, hist, los = k_fold(fun,input,target,opti,ner,epo)
                podatki['models'].append(model)
                podatki['hist'].append(hist.history['val_loss'])
                podatki['los'].append(los)
                podatki['epochs'].append(epo)
                podatki['nerv'].append(ner)
                podatki['opti'].append(opti)

    return podatki

def E(X_n,X_r):
    s = np.average((X_n-X_r)**2)
    return np.sqrt(s)



###UREJANJE PODATKOV###

# load data
data = np.load('data_integrator.npy')
# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(data)
# apply transform
normalized = scaler.transform(data)

print(len(data))
Nend = 997000 # velikost mnozice za ucenje in validacijo

X,Y=normalized[:Nend],normalized[1:Nend+1]
#X,Y = split_sequences(normalized[:Nend],time_steps_predictor)
new_order = np.random.choice(range(X.shape[0]),X.shape[0],replace=False)

X = X[new_order]
Y = Y[new_order]
print(X.shape)


###FITANJE MODELA###


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, verbose=0, mode='min')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5, min_lr=0.0001)
if True:
    model = make_model()

    history = model.fit(X, Y, epochs=200, batch_size=1024,validation_split=0.2,callbacks=[early_stop,reduce_lr])
    plot_train_history(history,'loss_plot_gosto_povzana')

    Y_predicted = np.zeros((1000,3))
    Y_predicted[0] = normalized[Nend+1]
    for z in range(0,999):
       Y_predicted[z+1]=model.predict(Y_predicted[z].reshape((1,3)))

    plot_series(data[Nend+1:Nend+1000],scaler.inverse_transform(Y_predicted[1:]),'series_plot_gosto_povzana')


###tri NN##

if True:
    #X,Y = split_sequences(normalized[:Nend],time_steps_predictor)
    new_order = np.random.choice(range(X.shape[0]),X.shape[0],replace=False)

    X = X[new_order]
    Y = Y[new_order]

    model1 = make_model(input_dim=1)
    model2 = make_model(input_dim=1)
    model3 = make_model(input_dim=1)

    history = model1.fit(X, Y[:,0], epochs=200, batch_size=1024,validation_split=0.2,callbacks=[early_stop,reduce_lr])
    history = model2.fit(X, Y[:,1], epochs=200, batch_size=1024,validation_split=0.2,callbacks=[early_stop,reduce_lr])
    history = model3.fit(X, Y[:,2], epochs=200, batch_size=1024,validation_split=0.2,callbacks=[early_stop,reduce_lr])

    Y_predicted = np.zeros((1000,3))
    Y_predicted[0] = normalized[Nend+1]
    for z in range(0,999):
       Y_predicted[z+1,0]=model1.predict([Y_predicted[z].reshape(1,3)])[0][0]
       Y_predicted[z+1,1]=model2.predict([Y_predicted[z].reshape(1,3)])[0][0]
       Y_predicted[z+1,2]=model3.predict([Y_predicted[z].reshape(1,3)])[0][0]

    plot_series(data[Nend+1:Nend+1000],scaler.inverse_transform(Y_predicted[1:]),'series_plot_gosto_povzana_vsaka_posebej')


####RNN######

time_steps_predictor = 16 # dolzina vhodnega zaporedja

X_rnn,Y_rnn = split_sequences(normalized[:Nend],time_steps_predictor)
new_order = np.random.choice(range(X_rnn.shape[0]),X_rnn.shape[0],replace=False)


X_rnn = X_rnn[new_order]
Y_rnn = Y_rnn[new_order]
'''
params = dict(
    opti= ['Adam','SGD','RMSprop','Adamax'],
    nerv = [10,16,25,60],
    epochs = [30])
'''

rezult = dict(
    opti= ['Adam'],
    nerv = [60],
    epochs = [30],
    num = [2])

best = 0
best_epochs = rezult['epochs'][best]
best_ner = rezult['nerv'][best]
best_opti = rezult['opti'][best]
best_num = rezult['num'][best]
model_rnn = model_RNN(best_ner,best_opti,time_steps_predictor,best_num)
history = model_rnn.fit(X_rnn, Y_rnn, epochs=best_epochs, validation_split=0.3)
plot_train_history(history,'loss_plot_RNN')

Y_predicted_rnn = predict(model_rnn,1000,normalized[Nend+1:Nend+17])

skupaj = np.zeros((3,1000,3))

skupaj[0] = scaler.inverse_transform(Y_predicted_rnn)
model_rnn.save('model_rnn')
plot_series(data[Nend+17:Nend+1001],scaler.inverse_transform(Y_predicted_rnn[16:]),'series_plot_RNN')

####LSTM######
model_lstm = model_LSTM(best_ner, best_opti,time_steps_predictor,best_num)
history = model_lstm.fit(X_rnn, Y_rnn, epochs=best_epochs, validation_split=0.3)
plot_train_history(history,'loss_plot_LSTM')

Y_predicted_LSTM = predict(model_lstm,1000,normalized[Nend+1:Nend+17])
skupaj[1] = scaler.inverse_transform(Y_predicted_LSTM)
model_lstm.save('model_lstm')
plot_series(data[Nend+17:Nend+1001],scaler.inverse_transform(Y_predicted_LSTM[16:]),'series_plot_LSTM')

####GRU######
model_gru = model_GRU(best_ner, best_opti ,time_steps_predictor,best_num)
history = model_gru.fit(X_rnn, Y_rnn, epochs=best_epochs, validation_split=0.3)
plot_train_history(history,'loss_plot_GRU')

Y_predicted_GRU = predict(model_gru,1000,normalized[Nend+1:Nend+17])
skupaj[2] = scaler.inverse_transform(Y_predicted_GRU)
model_gru.save('model_gru')
plot_series(data[Nend+17:Nend+1001],scaler.inverse_transform(Y_predicted_GRU[16:]),'series_plot_GRU')

plot_series(data[Nend+17:Nend+1001],skupaj[:,16:,:],'series_plot_skupaj',nn=['RNN','LSTM','GRU'])

##LSTM---one--to----one####
time_steps_predictor = 1 # dolzina vhodnega zaporedja

X_one,Y_one = split_sequences(normalized[:Nend],time_steps_predictor)
new_order = np.random.choice(range(X_one.shape[0]),X_one.shape[0],replace=False)

X_one = X_one[new_order]
Y_one = Y_one[new_order]

model_lstm1 = model_LSTM(best_ner, best_opti,time_steps_predictor,best_num)
history = model_lstm1.fit(X_one, Y_one, epochs=best_epochs, validation_split=0.3)
plot_train_history(history,'loss_plot_LSTM')

Y_predicted_LSTM = predict(model_lstm1,1000,normalized[Nend+1])

plot_series(data[Nend+1:Nend+1000],scaler.inverse_transform(Y_predicted_LSTM[1:]),'series_plot_LSTM_one_to_one')

####diferenca######

#podatki

defi = data[1:] - data[:-1]

scaler_defi = MinMaxScaler()
scaler_defi.fit(defi)
normalized_defi = scaler_defi.transform(defi)

X,Y=normalized[:Nend],normalized_defi[:Nend]
new_order = np.random.choice(range(X.shape[0]),X.shape[0],replace=False)

X = X[new_order]
Y = Y[new_order]

model = make_model()

history = model.fit(X, Y, epochs=200, batch_size=1024,validation_split=0.2,callbacks=[early_stop,reduce_lr])
plot_train_history(history,'loss_plot_gosto_povzana_plus_defi')


Y_predicted = np.zeros((1000,3))
Y_predicted[0] = data[Nend+1]
for z in range(0,999):
    za_napoved = scaler.transform([Y_predicted[z]])[0]
    Y_predicted[z+1] = scaler_defi.inverse_transform(model.predict(za_napoved.reshape((1,3))))[0]+Y_predicted[z]

plot_series(data[Nend+1:Nend+1000],Y_predicted[1:],'series_plot_gosto_povzana_gradient')

####SLABI ZAČETNI POGOJI######

time_steps_predictor = 16

data_zero = np.load('data_integrator_near_zero_init.npy')
data_far = np.load('data_integrator_far_from_zero_init.npy')

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
# fit scaler on data
scaler1.fit(data_zero)
scaler2.fit(data_far)
# apply transform
normalized1 = scaler1.transform(data_zero)
normalized2 = scaler2.transform(data_far)

data_zero_int = normalized1[:time_steps_predictor]
data_far_int = normalized2[:time_steps_predictor]


data_zero_rnn  = predict(model_rnn,1000,data_zero_int)
data_zero_lstm  = predict(model_lstm,1000,data_zero_int)
data_zero_gru  = predict(model_gru,1000,data_zero_int)

data_far_rnn  = predict(model_rnn,1000,data_far_int)
data_far_lstm  = predict(model_lstm,1000,data_far_int)
data_far_gru  = predict(model_gru,1000,data_far_int)

data_far_rnn  = scaler2.inverse_transform(data_far_rnn)
data_far_lstm  = scaler2.inverse_transform(data_far_lstm)
data_far_gru  = scaler2.inverse_transform(data_far_gru)
data_zero_rnn  = scaler1.inverse_transform(data_zero_rnn)
data_zero_lstm  = scaler1.inverse_transform(data_zero_lstm)
data_zero_gru  = scaler1.inverse_transform(data_zero_gru)

plot_series(data_zero[:1000],[data_zero_rnn,data_zero_lstm,data_zero_gru ],'series_plot_skupaj_zero',nn=['RNN','LSTM','GRU'])
plot_series(data_zero[:1000],[data_far_rnn,data_far_lstm,data_far_gru ],'series_plot_skupaj_far',nn=['RNN','LSTM','GRU'])

data_far_rnn=data_far_rnn-data_far[:1000]
data_far_lstm=data_far_lstm-data_far[:1000]
data_far_gru=data_far_gru-data_far[:1000]
data_zero_rnn=data_zero_rnn-data_zero[:1000]
data_zero_lstm=data_zero_lstm-data_zero[:1000]
data_zero_gru=data_zero_gru-data_zero[:1000]


final_rnn = np.sqrt(data_far_rnn[:,0]**2+data_far_rnn[:,1]**2+data_far_rnn[:,2]**2)
final_lstm = np.sqrt(data_far_lstm[:,0]**2+data_far_lstm[:,1]**2+data_far_lstm[:,2]**2)
final_gru = np.sqrt(data_far_gru[:,0]**2+data_far_gru[:,1]**2+data_far_gru[:,2]**2)
final_rnn1 = np.sqrt(data_zero_rnn[:,0]**2+data_zero_rnn[:,1]**2+data_zero_rnn[:,2]**2)
final_lstm1 = np.sqrt(data_zero_lstm[:,0]**2+data_zero_lstm[:,1]**2+data_zero_lstm[:,2]**2)
final_gru1= np.sqrt(data_zero_gru[:,0]**2+data_zero_gru[:,1]**2+data_zero_gru[:,2]**2)

s = ['solid','dashed','dashdot','--','-.']

t = len(final_gru1)
figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
plt.plot(np.linspace(0,t*0.01,t), final_rnn1, 'b', label='RNN',linestyle=s[0])
plt.plot(np.linspace(0,t*0.01,t), final_lstm1, 'r', label='LSTM',linestyle=s[1])
plt.plot(np.linspace(0,t*0.01,t), final_gru1, 'y', label='GRU',linestyle=s[2])
plt.xlabel('time')
plt.xlabel('loss')
plt.legend()
plt.savefig('Slike/residula_zero.png')


figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
plt.plot(np.linspace(0,t*0.01,t), final_rnn, 'b', label='RNN',linestyle=s[0])
plt.plot(np.linspace(0,t*0.01,t), final_lstm, 'r', label='LSTM',linestyle=s[1])
plt.plot(np.linspace(0,t*0.01,t), final_gru, 'y', label='GRU',linestyle=s[2])
plt.xlabel('time')
plt.xlabel('loss')
plt.legend()
plt.savefig('Slike/residula_far.png')



##dalsi koraka###
def dalsi(n):
    time_steps_predictor = 16 # dolzina vhodnega zaporedja

    X_rnn_long,Y_rnn_long = split_sequences(normalized[:Nend],time_steps_predictor)


    Y_rnn_long = Y_rnn_long[n:]
    X_rnn_long = Y_rnn_long[:n]

    new_order = np.random.choice(range(X_rnn_long.shape[0]),X_rnn_long.shape[0],replace=False)

    X_rnn_long = X_rnn_long[new_order]
    Y_rnn_long = Y_rnn_long[new_order]


    model_rnn_long = model_LSTM(best_ner, best_opti ,time_steps_predictor,best_num)

    history = model_rnn_long.fit(X_rnn_long, Y_rnn_long, epochs=best_epochs, validation_split=0.3)
    plot_train_history(history,'loss_plot_RNN_long'+str(n))


    Y_predicted_rnn = predict(model_rnn_long,1000,normalized[Nend+1:Nend+17+n],n=n)

    plot_series(data[Nend+17+n:Nend+1001],scaler.inverse_transform(Y_predicted_rnn[16+n:]),'series_plot_RNN_long_steps'+str(n))
    return data[Nend+17+n:Nend+1001],scaler.inverse_transform(Y_predicted_rnn[16+n:])

y2,d2 = dalsi(2)
y5,d5 = dalsi(5)
y10,d10 = dalsi(10)
y50,d50 = dalsi(50)
pod = [(y2,d2),(y5,d5),(y10,d10),(y50,d50)]
labe=['2 koraka naprej','5 koraka naprej','10 koraka naprej','50 koraka naprej']

figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
for stev in range(4):
    z, f = pod[stev]
    En = [E(z[:h],f[:h]) for h in range(900)]
    t=900
    plt.plot(np.linspace(0,t*0.01,t), En, label=labe[stev],linestyle=s[stev])

plt.xlabel('time')
plt.xlabel('E(t)')
plt.legend()
plt.savefig('Slike/E.png')
