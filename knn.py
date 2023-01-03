# useful libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd


###############################################################################

#  .------------------.  #
#  .    FUNCTIONS     .  #
#  .------------------.  #

def read_trainDataset():
    return pd.read_csv("../input/ghoulsgoblinsghosts/train.csv")

def read_testDataset():
    return pd.read_csv("../input/ghoulsgoblinsghosts/test.csv")

def remove_ids():
    ret = testDataset['id']
    del trainDataset["id"], testDataset["id"]
    return ret

def replace_color_values_trainDataset(trainDataset):
    nonDuplicates = []
    for i in trainDataset["color"]:
        if (i not in nonDuplicates):
            nonDuplicates.append(i)
    
    cnt = 1
    for value in nonDuplicates:
        trainDataset["color"].replace([value], cnt , inplace=True)
        cnt = cnt + 1
        
def replace_color_values_testDataset(testDataset):
    nonDuplicates = []
    for i in testDataset["color"]:
        if (i not in nonDuplicates):
            nonDuplicates.append(i)
    
    cnt = 1
    for value in nonDuplicates:
        testDataset["color"].replace([value], cnt , inplace=True)
        cnt = cnt + 1

def split_trainDataset(trainDataset):
    return trainDataset.iloc[:,:5], trainDataset.iloc[:,5:], trainDataset.iloc[:,:5], trainDataset.iloc[:,5:]

###############################################################################

#  .------------------.  #
#  .      DATA        .  #
#  .------------------.  #

# reading the data
trainDataset = read_trainDataset()
testDataset  = read_testDataset()

# removing ids from both trainDataset and testDataset
id_from_testDataset = remove_ids()

# changing the categorical values of 'color' column to integer values
replace_color_values_trainDataset(trainDataset)
replace_color_values_testDataset(testDataset)

# spliting trainDataset to sub-datasets
train_X, train_y, test_X, test_y = split_trainDataset(trainDataset)

###############################################################################

#  .------------------.  #
#  . [METHOD 1]  KNN  .  #
#  .------------------.  #

# models for k = 1, 3, 5, 10
knn1 = KNeighborsClassifier(n_neighbors = 1)
knn2 = KNeighborsClassifier(n_neighbors = 3)
knn3 = KNeighborsClassifier(n_neighbors = 5)
knn4 = KNeighborsClassifier(n_neighbors = 10)

# model training
knn1.fit(train_X, train_y.values.ravel())
knn2.fit(train_X, train_y.values.ravel())
knn3.fit(train_X, train_y.values.ravel())
knn4.fit(train_X, train_y.values.ravel())

# model predicting
yPred_test1, yPred_train1 = knn1.predict(testDataset), knn1.predict(test_X)
yPred_test2, yPred_train2 = knn2.predict(testDataset), knn2.predict(test_X)
yPred_test3, yPred_train3 = knn3.predict(testDataset), knn3.predict(test_X)
yPred_test4, yPred_train4 = knn4.predict(testDataset), knn4.predict(test_X)

# calculating accuracy and f1 score for all models
ACC_1 = accuracy_score(test_y, yPred_train1)
F1_1 = f1_score(test_y, yPred_train1, average='weighted')

ACC_2 = accuracy_score(test_y, yPred_train2)
F1_2 = f1_score(test_y, yPred_train2, average='weighted')

ACC_3 = accuracy_score(test_y, yPred_train3)
F1_3 = f1_score(test_y, yPred_train3, average='weighted')

ACC_4 = accuracy_score(test_y, yPred_train4)
F1_4 = f1_score(test_y, yPred_train4, average='weighted')

###############################################################################

#  .------------------.  #
#  .     PRINTS       .  #
#  .------------------.  #
print('--> k = 1 <--')
print("Accuracy =", format(ACC_1 * 100) , "% and F1 score = {}\n".format(F1_1))

print('--> k = 3 <--')
print("Accuracy =", format(ACC_2 * 100) , "% and F1 score = {}\n".format(F1_2))

print('--> k = 5 <--')
print("Accuracy =", format(ACC_3 * 100) , "% and F1 score = {}\n".format(F1_3))

print('--> k = 10 <--')
print("Accuracy =", format(ACC_4 * 100) , "% and F1 score = {}\n".format(F1_4))

###############################################################################

#  .------------------.  #
#  .   SUBMISSIONS    .  #
#  .------------------.  #
submit_a = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test1})
submit_a.to_csv('submit_a.csv', index=False)

submit_b = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test2})
submit_b.to_csv('submit_b.csv', index=False)

submit_c = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test3})
submit_c.to_csv('submit_c.csv', index=False)

submit_d = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test4})
submit_d.to_csv('submit_d.csv', index=False)