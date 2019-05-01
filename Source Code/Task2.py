import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.model_selection import train_test_split

batch_size = 60
nb_classes = 10
nb_epoch = 11

dataset = pd.read_csv("C:/Users/vinee/Downloads/heart.csv", header=None).values
 #print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[1:,0:13], dataset[1:,13],
                                                    test_size=0.25, random_state=87)

print(X_train)
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
X_train /= 255
X_test /= 255
Y_Train = np_utils.to_categorical(Y_train, nb_classes)
Y_Test = np_utils.to_categorical(Y_test, nb_classes)

#Logistic_regression
model = Sequential()
model.add(Dense(output_dim=10, input_shape=(13,), init='normal', activation='softmax'))
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#tensorboard graph genertion
tensorboard = TensorBoard(log_dir="logslo1/{}",histogram_freq=0, write_graph=True, write_images=True)
history=model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size,callbacks=[tensorboard])

#predicting accuracy
score = model.evaluate(X_test, Y_Test, verbose=1)
print('Loss: %.2f, Accuracy: %.2f' % (score[0], score[1]))

#plotting loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()