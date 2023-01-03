# useful libraries
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
from keras.layers import Flatten, Dense, Activation
from keras.models import Sequential
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
    hold_type = pd.get_dummies(trainDataset["type"])
    trainDataset.drop("type", axis=1)

    return trainDataset.iloc[:,:5], hold_type, trainDataset.iloc[:,:5], hold_type

def replace_type_values_submbit(submit_b):
    nonDuplicates = []
    for i in submit_b["type"]:
        if (i not in nonDuplicates):
            nonDuplicates.append(i)
    
    for value in nonDuplicates:
        if (value == 1):
            submit_b["type"].replace([value], "Ghoul" , inplace=True)
        elif (value == 2):
            submit_b["type"].replace([value], "Goblin" , inplace=True)
        else:
            submit_b["type"].replace([value], "Ghost" , inplace=True)

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

#  .------------------------------.  #
#  . [METHOD 2]  Neural Networks  .  #
#  .------------------------------.  #
# there are 5 columns as input shape for the neural network
inputShape = train_X.shape[1]

# info
K1, K2 = 100, 50
exit = 3
modelCompile = ['accuracy', 'categorical_crossentropy', 'sgd']

NeuralNetSeq2 = Sequential()
NeuralNetSeq2.add(Flatten(input_shape=(inputShape,)))
NeuralNetSeq2.add(Dense(K1, activation = "sigmoid"))
NeuralNetSeq2.add(Dense(K2, activation = "sigmoid"))
NeuralNetSeq2.add(Dense(exit, activation = "softmax"))

# model compiling
NeuralNetSeq2.compile(metrics = [modelCompile[0]], loss = modelCompile[1], optimizer = modelCompile[2])

# model training
NeuralNetSeq2.fit(train_X, train_y, batch_size=15, epochs=15)

# model predicting
yPred_test2, yPred_train2 = NeuralNetSeq2.predict(testDataset), NeuralNetSeq2.predict(test_X)

# downcasting yPred_train1, yPred_test1 to 1 dimension because test_y needs to be numpy 1d array
yPred_train2, yPred_test2 = np.argmax(yPred_train2, axis=1), np.argmax(yPred_test2, axis=1)
test_y = np.argmax(test_y.values, axis=1)

# calculating accuracy and f1 score
ACC_2 = accuracy_score(test_y, yPred_train2)
F1_2 = f1_score(test_y, yPred_train2, average='weighted')

###############################################################################

#  .------------------.  #
#  .     PRINTS       .  #
#  .------------------.  #
print('\n\n--> 2 κρυμμένα επίπεδα αποτελούμενο από Κ1 = 100 και Κ2 = 50 νευρώνες <--')
print("Accuracy =", format(ACC_2 * 100) , "% and F1 score = {}\n".format(F1_2))

###############################################################################

#  .------------------.  #
#  .   SUBMISSIONS    .  #
#  .------------------.  #
submit_b = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test2})
replace_type_values_submbit(submit_b)
submit_b.to_csv('submit_b.csv', index=False)