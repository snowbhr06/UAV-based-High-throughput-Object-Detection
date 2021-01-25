# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:24:49 2021

@author: Administrator
"""


from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score
import scipy.io



dataset = scipy.io.loadmat('train_data.mat')
train_data = dataset['train_data']
X_train = train_data[:,0:-1]
y_train = train_data[:,-1]

dataset = scipy.io.loadmat('test_data.mat')
test_data = dataset['test_data']
X_test = test_data[:,0:-1]
y_test = test_data[:,-1]

# split data into train and test sets
seed = 7
test_size = 0.33

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# save model to file
pickle.dump(model, open("plant.pickle.dat", "wb"))
print("Saved model to: plant.pickle.dat")
# some time later...
# load model from file
loaded_model = pickle.load(open("plant.pickle.dat", "rb"))
print("Loaded model from: plant.pickle.dat")
# make predictions for test data
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))