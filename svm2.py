# useful libraries
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
###############################################################################

#  .------------------.  #
#  .    FUNCTIONS     .  #
#  .------------------.  #

def read_trainDataset():
    return pd.read_csv("/kaggle/input/ghoulsgoblinsghosts/train.csv")

def read_testDataset():
    return pd.read_csv("/kaggle/input/ghoulsgoblinsghosts/test.csv")

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
    return trainDataset.iloc[:,:5], trainDataset.iloc[:,5:].type.values, trainDataset.iloc[:,:5], trainDataset.iloc[:,5:].type.values

###############################################################################

#  .------------------.  #
#  .      DATA        .  #
#  .------------------.  #

# reading the data
trainDataset = read_trainDataset()
testDataset  = read_testDataset()

# removig ids from both trainDataset and testDataset
id_from_testDataset = remove_ids()

# changing the categorical values of 'color' column to integer values
replace_color_values_trainDataset(trainDataset)
replace_color_values_testDataset(testDataset)

# spliting trainDataset to sub-datasets
train_X, train_y, test_X, test_y = split_trainDataset(trainDataset)

###############################################################################

#  .---------------------------------------.  #
#  . [METHOD 3]  Support Vector Machines   .  #
#  .---------------------------------------.  #

# info
infos = ["ovr", "rbf"]
C1, gamma1 = 0.05, 1
C2, gamma2 = 1, 0.05
C3, gamma3 = 0.5, 1.1
C4, gamma4 = 1.1, 0.5

# models
clf1 = SVC(decision_function_shape = infos[0], kernel = infos[1], gamma = gamma1, C = C1)
clf2 = SVC(decision_function_shape = infos[0], kernel = infos[1], gamma = gamma2, C = C2)
clf3 = SVC(decision_function_shape = infos[0], kernel = infos[1], gamma = gamma3, C = C3)
clf4 = SVC(decision_function_shape = infos[0], kernel = infos[1], gamma = gamma4, C = C4)

# model training
clf1.fit(train_X, train_y)
clf2.fit(train_X, train_y)
clf3.fit(train_X, train_y)
clf4.fit(train_X, train_y)

# model predicting
yPred_test1, yPred_train1 = clf1.predict(testDataset), clf1.predict(test_X)
yPred_test2, yPred_train2 = clf2.predict(testDataset), clf2.predict(test_X)
yPred_test3, yPred_train3 = clf3.predict(testDataset), clf3.predict(test_X)
yPred_test4, yPred_train4 = clf4.predict(testDataset), clf4.predict(test_X)

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
print('\n\n--> Support Vector Machines <--')
print("Gaussian συνάρτηση πυρήνα RBF (kernel)")

print("[gamma = " + format(gamma1) + ", C = " + format(C1) + "]")
print("Accuracy =", format(ACC_1 * 100) , "% and F1 score = {}\n".format(F1_1))

print("[gamma = " + format(gamma2) + ", C = " + format(C2) + "]")
print("Accuracy =", format(ACC_2 * 100) , "% and F1 score = {}\n".format(F1_2))

print("[gamma = " + format(gamma3) + ", C = " + format(C3) + "]")
print("Accuracy =", format(ACC_3 * 100) , "% and F1 score = {}\n".format(F1_3))

print("[gamma = " + format(gamma4) + ", C = " + format(C4) + "]")
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