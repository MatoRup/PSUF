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




def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.savefig('Slike/2'+title+'.png')
    plt.show()

def TF_model(optimizer='adam', neurons1=200,neurons2=100, learning_rate=0.001):
    model = Sequential ()
    model.add(Dense (neurons1, activation ='sigmoid', input_shape =(400,)))
    model.add(Dense (neurons2, activation ='sigmoid'))
    model.add(Dense (2, activation ='sigmoid'))

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile( optimizer=opt, loss='mean_absolute_error', metrics =(['mean_squared_error']))

    return model


lam = ['500','505','700','900']
DATA = np.zeros((4,100000, 400))

for j in range(4):
    DATA[j] = np.load('DataK13_pp/intensity'+lam[j]+'noise0.npy')
Kvalues = np.load('DataK13_pp/Kvalues.npy')
theta=  np.load('DataK13_pp/theta0.npy')

figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
plt.plot(np.linspace(0,1.20,400),DATA[0,0,:])
plt.ylabel('$I(t)$')
plt.xlabel('$t$')
plt.savefig('Slike/example2k.png')
plt.show()

predic_star = 90000
MaxK = np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK



X_train, X_test, y_train, y_test = train_test_split( DATA[0,:predic_star,:], Kvalues_norm[:predic_star], test_size=0.20, random_state=42)

model = KerasClassifier(model=TF_model(), loss="binary_crossentropy", epochs=30, batch_size=10, verbose=0)


#batches = [10, 30, 40,60,100]
#epoch_values = [5,10, 20,50,60]
#optimizers = ['adam', 'SGD']
#neuron1_list = [100,250, 200,300]
#neuron2_list = [30, 50,100,200]
#lr_values = [0.001,0.01,0.1,1 ]

batches = [10, 30]
epoch_values = [10, 20]
optimizers = ['adam', 'SGD']
neuron1_list = [250, 200]
neuron2_list = [250, 200]
lr_values = [0.001 ]


param_grid = dict(
    epochs=epoch_values,
    batch_size=batches,
    optimizer=optimizers,
    neurons1=neuron1_list,
    neurons2=neuron1_list,
    learning_rate=lr_values)



grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)


grid_result = grid.fit(X_train,y_train, validation_data=(X_test,y_test))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

X_predict = DATA[0,predic_star:,:]
y_predict_rela = Kvalues_norm[predic_star:]

plot_train_history(History, 'Multi-Step Training and validation loss')

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

res = grid_result.best_params_

for j in ['adam','SGD']:
    model = TF_model(optimizer=j, neurons1=res['neurons2'],neurons2=res['neurons2'], learning_rate=res['learning_rate'])
    History = model.fit(X_train,y_train, epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(X_test,y_test))
    plot_train_history(History, 'Multi-Step Training and validation loss')
    y_predict_predic = model.predict(X_predict)
    figure = plt.subplots(figsize=(10, 6),constrained_layout=True)
    plt.scatter(y_predict_rela*MaxK,y_predict_predic*MaxK,s=1)
    plt.plot([0,2e-11],[0,2e-11], c='r')
    plt.ylabel('$K_{predvideni}$')
    plt.xlabel('$K_{pravi}$')
    plt.ylim([0, 2.3e-11])
    plt.savefig('Slike/'+'BestGVresultK1'+j+'.png')
    plt.show()

####K-fold cross-validation####

X = DATA[0,:predic_star,:]
y = Kvalues_norm[:predic_star]

for kfold, (train, test) in enumerate(KFold(n_splits=10, shuffle=True).split(X, y)):

    # clear the session
    tf.keras.backend.clear_session()

    # calling the model and compile it
    model = TF_model(optimizer=j, neurons1=res['neurons2'],neurons2=res['neurons2'], learning_rate=res['learning_rate'])

    plot_train_history(History, 'Multi-Step Training and validation loss')

    print('Train Set')
    print(X[train].shape)
    print(y[train].shape)

    print('Test Set')
    print(X[test].shape)
    print(y[test].shape)

    # run the model
    model.fit(X_train,y_train, epochs=res['epochs'], batch_size=res['batch_size'], validation_data=(X_test,y_test))
    eval = model.evaluate(X[test], y[test])
    model.save_weights(f'weightK\2kwg_{kfold}s'+str(eval)+'.h5')


y_predict_predic =seq_model.predict(X_predict)


predic_star = 90000
MaxK = np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK
