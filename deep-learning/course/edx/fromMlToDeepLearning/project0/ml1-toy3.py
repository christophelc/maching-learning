import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#not working well:
#Liblinear failed to converge, increase the number of iterations

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

fig, ax = plt.subplots()
ax.matshow(digits.images[0])
X_train.shape
#plt.show()

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

clf = Perceptron(max_iter=40, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

clf = LinearSVC(C=1, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, clf.predict(X_test))

clf = LinearSVC(C=1, random_state=0)
from sklearn.model_selection import cross_val_score
scores =  cross_val_score(clf, X_train, y_train, cv=5)
print("Mean: %.4f, Std: %.4f" % (np.mean(scores), np.std(scores)))

clf = LinearSVC(C=0.1, random_state=0)
scores =  cross_val_score(clf, X_train, y_train, cv=5)
print("Mean: %.4f, Std: %.4f" % (np.mean(scores), np.std(scores)))

#Cross validation
from sklearn.model_selection import GridSearchCV
clf = LinearSVC(random_state=0)
param_grid = {'C': 10. ** np.arange(-6, 4)}
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=3, return_train_score=True)

print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred = grid_search.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

