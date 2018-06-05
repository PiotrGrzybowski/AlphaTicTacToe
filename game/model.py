import matplotlib.pyplot as plt
import numpy as np

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    X = np.load('X.npy')
    Y = np.load('Y.npy')

    clf = Ridge(alpha=1.0)
    clf.fit(X, Y)

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # # define and fit the final model
    # model = Sequential()
    # model.add(Dense(10, input_dim=9, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(5, input_dim=9, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    # model.compile(loss='mse', optimizer='adam')
    #
    # history = model.fit(X, Y, epochs=300, verbose=1, validation_split=0.05)
    # # prediction = model.predict(X)
    # # # show the inputs and predicted outputs
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved model to disk")
    #
    # print(history.history.keys())
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Mean Squared Error')
    # plt.ylim(0, 0.5)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
