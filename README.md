Implementing popular AI classification algorithms like Nearest Neighbor kNN, Neural Networks and Support Vector Machines (SVM).

To run the code and you have to load the competition dataset Ghouls, Goblins, and Ghosts... Boo! for more information about the dataset, visit the site https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo.

•	Nearest Neighbor kNN
Runs for the number of neighbors (k) : 1, 3, 5, 10.

•	Neural Networks (Sequential model)
Both cases use the sigmoid activation function and for the training they use Stochastic Gradient Descent.

a) with 1 hidden layer and 100 hidden neurons
b) with 2 hidden layers and 100 and 50 neurons


• Support Vector Machines (SVM)
All cases use the one-versus-all strategy as the problem is a multi-class dataset.

a) Linear kernel
b) Gaussian kernel
c) Cosine kernel

• The evaluation of the performance of the methods can be computed by the Accuracy and and F1 score.

Accuracy : 100 * (TP + TN) / (P + N)

F1 score : 2 * (Precision * Recall) / (Precision + Recall)
where TP: true positives, TN: true negative, FN: false negative, FP: false positives, P: positives, N: negatives and Precision = TP/(TP + FP) and Recall = TP/(TP + FN).
