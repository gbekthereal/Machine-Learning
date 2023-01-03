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
infos = ["ovr", "linear"]

clf = SVC(decision_function_shape = infos[0], kernel = infos[1])

# model training
clf.fit(train_X, train_y)

# model predicting
yPred_test1, yPred_train1 = clf.predict(testDataset), clf.predict(test_X)

# calculating accuracy and f1 score
ACC_1 = accuracy_score(test_y, yPred_train1)
F1_1 = f1_score(test_y, yPred_train1, average='weighted')

###############################################################################

#  .------------------.  #
#  .     PRINTS       .  #
#  .------------------.  #
print('\n\n--> Support Vector Machines <--')
print("γραμμική συνάρτηση πυρήνα (linear kernel)")
print("Accuracy =", format(ACC_1 * 100) , "% and F1 score = {}\n".format(F1_1))

###############################################################################

#  .------------------.  #
#  .   SUBMISSIONS    .  #
#  .------------------.  #
submit_a = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test1})
submit_a.to_csv('submit_a.csv', index=False)