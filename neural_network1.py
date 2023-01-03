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

def replace_type_values_submbit(submit_a):
    nonDuplicates = []
    for i in submit_a["type"]:
        if (i not in nonDuplicates):
            nonDuplicates.append(i)
    
    for value in nonDuplicates:
        if (value == 1):
            submit_a["type"].replace([value], "Ghoul" , inplace=True)
        elif (value == 2):
            submit_a["type"].replace([value], "Goblin" , inplace=True)
        else:
            submit_a["type"].replace([value], "Ghost" , inplace=True)

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
K = 100
exit = 3
modelCompile = ['accuracy', 'categorical_crossentropy', 'sgd']

NeuralNetSeq1 = Sequential()
NeuralNetSeq1.add(Flatten(input_shape = (inputShape,)))
NeuralNetSeq1.add(Dense(K, activation = "sigmoid"))
NeuralNetSeq1.add(Dense(exit, activation = "softmax"))

# model compiling
NeuralNetSeq1.compile(metrics = [modelCompile[0]], loss = modelCompile[1], optimizer = modelCompile[2])

# model training
NeuralNetSeq1.fit(train_X, train_y, batch_size=15, epochs=15)

# model predicting
yPred_test1, yPred_train1 = NeuralNetSeq1.predict(testDataset), NeuralNetSeq1.predict(test_X)

# downcasting yPred_train1, yPred_test1 to 1 dimension because test_y needs to be numpy 1d array
yPred_train1, yPred_test1 = np.argmax(yPred_train1, axis=1), np.argmax(yPred_test1, axis=1)
test_y = np.argmax(test_y.values, axis=1)

# calculating accuracy and f1 score
ACC_1 = accuracy_score(test_y, yPred_train1)
F1_1 = f1_score(test_y, yPred_train1, average='weighted')

###############################################################################

#  .------------------.  #
#  .     PRINTS       .  #
#  .------------------.  #
print('\n\n--> 1 κρυμμένο επίπεδο και Κ = 100 κρυμμένους νευρώνες <--')
print("Accuracy =", format(ACC_1 * 100) , "% and F1 score = {}\n".format(F1_1))

###############################################################################

#  .------------------.  #
#  .   SUBMISSIONS    .  #
#  .------------------.  #
submit_a = pd.DataFrame({'id':id_from_testDataset, 'type':yPred_test1})
replace_type_values_submbit(submit_a)
submit_a.to_csv('submit_a.csv', index=False)