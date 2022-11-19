import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def plot_train_history(history, title):
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


lam = ['500','505','700','900']

DATA = np.zeros((4,100000, 400))

for j in range(4):
    DATA[j] = np.load('/DataK/intensity'+lam[j]+'noise0.npy')
Kvalues = np.load('/DataK/Kvalues.npy')
theta=  np.load('/DataK/theta0.npy')

plt.plot(range(len(DATA[0,0,:])),DATA[0,0,:])
plt.ylabel('I(t)')
plt.xlabel('t')
plt.show()

predic_star= 90000
MaxK=np.max(Kvalues)
Kvalues_norm=Kvalues/MaxK

X_train, X_test, y_train, y_test = train_test_split( DATA[0,:predic_star,:], Kvalues_norm[:90000], test_size=0.20, random_state=42)



model = Sequential ()
model.add(Dense (200 , activation ='sigmoid', input_shape =(400 ,)))
model.add(Dense (100 , activation ='sigmoid'))
model.add(Dense (1, activation ='sigmoid'))
model.compile( optimizer ='adam', loss='mean_absolute_error', metrics =(['mean_squared_error']))

#model.compile(optimazer=keras.optimizers.Adam(learning_rate=0.01),loss='mean_absolute_error', metrics=['mean_square_error'])

History = model.fit(X_train,y_train, epochs=50, batch_size=25, validation_data=(X_test,y_test),shuffle=True)
X_predict = DATA[0,predic_star:,:]
y_predict_rela = Kvalues_norm[predic_star:]

plot_train_history(History, 'Multi-Step Training and validation loss')

y_predict_predic = model.predict(X_predict)


plt.scatter(y_predict_rela*MaxK,y_predict_predic*MaxK,s=1)
plt.plot([0,2e-11],[0,2e-11], c='r')
plt.ylabel('K_predvideni')
plt.xlabel('K_pravi')
plt.ylim([0, 2.3e-11])
plt.show()
