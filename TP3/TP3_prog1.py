# Imports :

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import time

# =================================== Dataset ==========================================================

mnist = fetch_openml('mnist_784', as_frame=False)

#randomise Data et target
indices = np.random.randint(10000, size=10000)
data = mnist.data[indices]
target = mnist.target[indices]

# training set of 49000 (70% training set - 30 % test set)
xtrain, xtest, ytrain, ytest =train_test_split(data, target, train_size=int(len(data)*0.7))

# ============================================= SVC with liearn kernel =============================================
# TRAINING
clf = SVC(kernel='linear')
beginTrain =time.time()
clf.fit(xtrain, ytrain)
endTrain = time.time()

# PREDICTION
beginpred= time.time()
prediction = clf.predict(xtest)
endpred = time.time()

# METRICS
score = clf.score(xtest,ytest)
recall = metrics.recall_score(ytest, prediction, average ='macro')
precision = metrics.precision_score(ytest, prediction,  average='macro')
loss01 = metrics.zero_one_loss(ytest, prediction)
timetrain = endTrain - beginTrain
timePred = endpred - beginpred

print("The SVC model with <Linear kernel> has a score of :", score * 100, "%, and a precision = ",precision * 100,"% ", "with a training time = ", timetrain, " s." )


# ===================================================== Trying diffrent kernels : =================================================================

KernelScore =[]
KernelPrecision = []
KernelRecall = []
KernelLoss = []
KernelTimeTraining = []
KernelTimePrediction = []

def changeKernel (mykernel):
    
    # TRAINING :
    clf = SVC(kernel=mykernel)
    start1 =time.time()
    clf.fit(xtrain, ytrain)
    end1 = time.time()

    # PREDICTION
    start2= time.time()
    predict = clf.predict(xtest)
    end2 = time.time()

    # METRICS
    score = clf.score(xtest,ytest)
    recall = metrics.recall_score(ytest, predict, average ='macro')
    precision = metrics.precision_score(ytest, predict,  average='macro')
    loss01 = metrics.zero_one_loss(ytest, predict)
    timetrain = end1 - start1
    timePred = end2 - start2

    #SAVING
    KernelScore.append(score*100)
    KernelPrecision.append(precision*100)
    KernelRecall.append(recall)
    KernelLoss.append(loss01)
    KernelTimePrediction.append(timePred)
    KernelTimeTraining.append(timetrain)

    #PRINT
    print("This SVC model with a kernel = ", mykernel, "has a score of : ", score*100, "%.")
    print("4th image : prediction =",predict[3], "reel one = : ", ytest[3])
    print ("precision :", precision * 100, "%.")
    print ("recall  :",recall * 100, "%.")
    print ("zero-one_loss :",recall * 100, "%.")
    print( "training time :", timetrain, " s")
    print( "prediction time :", timePred, " s")
    print("\n")

# Test :
kernels = ('linear', 'poly', 'rbf', 'sigmoid')
for ker in kernels:
    changeKernel(ker)

# Results visualization :
fig, axarr = plt.subplots(6, sharex=True, figsize=(10,10))
axarr[0].scatter(range(4), KernelScore, c='red')
axarr[0].set_title('The SCV Model using diffrent kernels : linear, poly, rbf and sigmoid')
axarr[0].set_ylabel('Score (%)')
axarr[1].scatter(range(4), KernelPrecision, c='blue')
axarr[1].set_ylabel('Precision (%)')
axarr[2].scatter(range(4), KernelRecall, c='green')
axarr[2].set_ylabel('Recall ')
axarr[3].scatter(range(4), KernelLoss, c='purple')
axarr[3].set_ylabel('Zero-to-one Loss')
axarr[4].scatter(range(4), KernelTimeTraining, c='orange')
axarr[4].set_ylabel('Training Time(s)')
axarr[5].scatter(range(4), KernelTimePrediction)
axarr[5].set_ylabel('¨Prediction Time(s)')

plt.show()

# ====================================================== cost (c) variation =======================================================================

CostScore =[]
CostPrecision = []
CostRecall = []
CostLoss = []
CostTimeTraining = []
CostTimePrediction = []

def changeCost (c):
    
    # TRAINING :
    clf = SVC(kernel='rbf', C=c)
    start1 = time.time()
    clf.fit(xtrain, ytrain)
    end1 = time.time()

    # PREDICTION
    start2 = time.time()
    predict = clf.predict(xtest)
    end2 = time.time()

    # METRICS
    score = clf.score(xtest,ytest)
    recall = metrics.recall_score(ytest, predict, average ='macro')
    precision = metrics.precision_score(ytest, predict,  average='macro')
    loss01 = metrics.zero_one_loss(ytest, predict)
    timetrain = end1 - start1
    timePred = end2 - start2

    #SAVING
    CostScore.append(score*100)
    CostPrecision.append(precision*100)
    CostRecall.append(recall)
    CostLoss.append(loss01)
    CostTimePrediction.append(timePred)
    CostTimeTraining.append(timetrain)

    #PRINT
    print("This SVC Model, kernel rbf et avec un cost ", c, "a un score de ", score*100, "%.")
    print("4eme image : prédiction ",predict[3], "reel : ", ytest[3])
    print ("précision :", precision*100)
    print ("recall  :",recall*100)
    print ("zero-one_loss :",recall*100)
    print( "training time :", timetrain)
    print( "prediction time :", timePred)
    print()

# Test :
values_of_c = np.linspace(0.1,1, 5, endpoint=True)
for c in values_of_c :
    changeCost(c)
# Results visualization :
fig, axarr = plt.subplots(6, sharex=True, figsize=(10,10))
axarr[0].plot(values_of_c, CostScore)
axarr[0].set_title('The SVC model with different costs (c) :  0.1, 0.325, 055, 0.775, 1')
axarr[0].set_ylabel('Score (%)')
axarr[1].plot(values_of_c, CostPrecision)
axarr[1].set_ylabel('Precision (%)')
axarr[2].plot(values_of_c, CostRecall)
axarr[2].set_ylabel('Recall ')
axarr[3].plot(values_of_c, CostLoss)
axarr[3].set_ylabel('Zero-to-one Loss')
axarr[4].plot(values_of_c, CostTimeTraining)
axarr[4].set_ylabel('Training Time in sec')
axarr[5].plot(values_of_c, CostTimePrediction)
axarr[5].set_ylabel('¨Prediction Time in sec')

plt.show()


# ============================================ matrice de confusion ============================================================
 

from sklearn.metrics import confusion_matrix

clasifier = SVC(kernel='linear')

begin = time.time()
clasifier.fit(xtrain,ytrain)
predicted = clasifier.predict(X=xtest)
end = time.time()

print("score: ", clasifier.score(xtest, ytest))

total_time = end - begin
print("time: ", total_time)

print(confusion_matrix(ytest,predicted))



# end .................
