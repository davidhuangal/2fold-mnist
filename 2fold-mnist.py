import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.datasets import load_digits

### DATA SETUP
digits = load_digits()
features, labels = digits.data, digits.target

# One-hot encoding the labels
labels = keras.utils.to_categorical(labels, num_classes=10)

# Randomizing the order of the data
p = numpy.random.permutation(len(features))
features = features[p]
labels = labels[p]

# Cross-Validation
num_samples = len(features)
cutoff = num_samples // 2

fold1_train, fold2_train = features[:cutoff], features[cutoff:]
fold1_train_labels, fold2_train_labels = labels[:cutoff], labels[cutoff:]

fold1_test, fold2_test = fold2_train, fold1_train
fold1_test_labels, fold2_test_labels = fold2_train_labels, fold1_train_labels


dimension = 64
num_classes = 10

##########################

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=dimension))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


##########################
from keras.models import clone_model
model_1 = clone_model(model)
model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_1.fit(fold1_train, fold1_train_labels, epochs=12)

model_2 = clone_model(model)
model_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_2.fit(fold2_train, fold2_train_labels, epochs=12)

##########################
loss, acc1 = model_1.evaluate(fold1_test, fold1_test_labels)
loss, acc2 = model_2.evaluate(fold2_test, fold2_test_labels)


print("Fold 1 {}".format(acc1))
print("Fold 2 {}".format(acc2))
