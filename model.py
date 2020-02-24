from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy


dataset = loadtxt('wine.csv', delimiter=',')
numpy.random.shuffle(dataset)
X = dataset[:,1:]
y = dataset[:,0]

y = to_categorical(y)

model = Sequential()
model.add(Dense(26, input_dim=13, activation='relu'))
model.add(Dense(18, input_dim=13, activation='relu'))
model.add(Dense(15, input_dim=13, activation='relu'))
model.add(Dense(4, activation='softmax'))


# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

history = model.fit(X, y, validation_split=0.4, epochs=300, batch_size=5)



# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


