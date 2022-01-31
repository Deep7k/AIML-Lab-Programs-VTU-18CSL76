"""Write a program to implement K-Nearest Neighbour algorithm to classify th iris data set, print both correct and
writing prediction. Java/Python library classes can be used for this problem"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Confusion matrix is as follows')
print(confusion_matrix(y_test, y_pred))
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))

########################################################################################################################
# OUTPUT:
# Ignore double quotes at beginning and end
########################################################################################################################

"""
Confusion matrix is as follows
[[10  0  0]
 [ 0  6  0]
 [ 0  2 12]]
Accuracy Metrics
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.75      1.00      0.86         6
           2       1.00      0.86      0.92        14

    accuracy                           0.93        30
   macro avg       0.92      0.95      0.93        30
weighted avg       0.95      0.93      0.94        30


Process finished with exit code 0
"""