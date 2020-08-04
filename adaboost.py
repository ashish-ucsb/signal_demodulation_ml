import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from scipy.io import loadmat
from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Prepare Data
x = loadmat('data/128qam_10p_0cm.mat')['data_10p_0cm'].T
y = loadmat('data/128qam_10p_label.mat')['org_label'][0]
print("Original Data: {}".format(x.shape))
print("Original Labels: {}".format(y.shape))
classes = len(unique_labels(y))

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
print("Train: {}, Test: {}".format(y_train.shape, y_test.shape))

class CustomClassifier(KNeighborsClassifier):
    def fit(self, x, y, sample_weight=None):
        return self._fit(x, y)

# Train
adaboost = AdaBoostClassifier(
	DecisionTreeClassifier(max_depth=5),
	n_estimators=100,
	learning_rate=1.5,
	algorithm="SAMME")

adaboost.fit(x_train, y_train)

# Predict
y_pred = adaboost.predict(x_test)

# Evaluation
print("classification Report:\n%s\n" % (
    metrics.classification_report(y_test, y_pred)))