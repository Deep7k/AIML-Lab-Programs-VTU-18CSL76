"""Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file.
Compute the accuracy of the classifier, considering few test data sets. """

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

msg = pd.read_csv('NBC.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
y = msg.labelnum

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names())
print(df[0:5])

clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)

for doc, p in zip(Xtrain, pred):
    p = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, p))

print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))

########################################################################################################################
# OUTPUT:
# Ignore single quotes at beginning and end
########################################################################################################################


'''
Total Instances of Dataset:  18
   about  am  amazing  an  awesome  beers  ...  today  very  view  went  what  work
0      0   0        0   1        1      0  ...      0     0     1     0     1     0
1      0   0        0   0        0      0  ...      0     0     0     0     0     0
2      0   0        0   0        0      0  ...      0     0     0     0     0     0
3      0   0        0   0        0      0  ...      1     0     0     1     0     0
4      0   0        0   1        1      0  ...      0     0     0     0     0     0

[5 rows x 39 columns]
What an awesome view -> pos
I love to dance -> neg
He is my sworn enemy -> pos
I went to my enemy's house today -> pos
This is an awesome place -> neg
Accuracy Metrics: 

Accuracy:  0.6
Recall:  1.0
Precision:  0.3333333333333333
Confusion Matrix: 
 [[2 2]
 [0 1]]

Process finished with exit code 0
'''
