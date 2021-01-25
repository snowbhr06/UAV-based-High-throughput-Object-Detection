# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:52:18 2021

@author: Administrator
"""


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from joblib import dump
import time

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=50),
    RandomForestClassifier(max_depth=50, n_estimators=100, max_features=10),
    MLPClassifier(alpha=1, max_iter=10000),
    AdaBoostClassifier(),
    GaussianNB()]

import scipy.io

dataset = scipy.io.loadmat('train_data.mat')
train_data = dataset['train_data']
X_train = train_data[:,0:-1]
y_train = train_data[:,-1]

dataset = scipy.io.loadmat('test_data.mat')
test_data = dataset['test_data']
X_test = test_data[:,0:-1]
y_test = test_data[:,-1]

# iterate over classifiers
scores = []
eval_times = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    start_time = time.time()
    score = clf.score(X_test, y_test)
    eval_time = time.time()-start_time
    eval_times.append(eval_time)
    scores.append(score)
    dump(clf,name+'.joblib')
    

