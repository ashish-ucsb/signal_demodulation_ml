from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# Prepare Data
x = loadmat('data/ook_10p_0cm.mat')['data_10p_0cm'].T
y = loadmat('data/ook_10p_label.mat')['org_label'][0]
classes = len(unique_labels(y))

print("Original Data: {}".format(x.shape))
print("Original Labels: {}".format(y.shape))
print("No. of Unique Classes: {}".format(classes))

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
print("Train: {}, Test: {}".format(y_train.shape, y_test.shape))

logistic = linear_model.LogisticRegression(C=100)
rbm1 = BernoulliRBM(n_components=50, learning_rate=0.01, n_iter=10, verbose=1, random_state=101)
rbm2 = BernoulliRBM(n_components=50, learning_rate=0.01, n_iter=10, verbose=1, random_state=101)
rbm3 = BernoulliRBM(n_components=200, learning_rate=0.01, n_iter=10, verbose=1, random_state=101)

dbn = Pipeline(steps=[('rbm1', rbm1),('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])

# Training RBM-Logistic Pipeline
dbn.fit(x_train, y_train)

# Evaluation
y_pred = dbn.predict(x_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, y_pred)))