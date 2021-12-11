#================================================# Multi-layered Neural Network MPL #=======================================================

# ----------------------------------Imports------------------------------------------------ :
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import time
import numpy as np

# --------------------------------------------------- Dataset -------------------------------------------
mnist = fetch_openml('mnist_784', as_frame=False)
indices = np.random.randint(70000, size=70000)
data = mnist.data[indices]
target = mnist.target[indices]
# split :
xtrain, xtest, ytrain, ytest =train_test_split(data, target, train_size=49000)

# ------------------------------------------------------ MLP with one Hidden Layer of 50 neurals --------------------------
clf = MLPClassifier(hidden_layer_sizes=(50))

clf.fit(xtrain, ytrain)
prediction = clf.predict(xtest)
score = clf.score(xtest, ytest)
recall = metrics.recall_score(ytest, prediction, average = 'macro')
precision = metrics.precision_score(ytest, prediction, average ='macro')
loss0_1 = metrics.zero_one_loss(ytest, prediction)

# Print & Test :------
print("This MLP model, with one layer of 50, has a score of : ", score*100, "%.")
print("4th image : Prediction ",prediction[3], "Vs  Reel : ", ytest[3])

# Showing the 4th predicted image:
images = xtest.reshape((-1, 28, 28))
plt.imshow(images[3],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()

# Metrics :
print ("This MLP model has a precision of :", precision*100, "%.")
print ("This MLP model has a recall of : ",recall*100, "%.")
print ("This MLP model has a zero-one_loss of :",recall*100, "%.")

# --------------------------- Variation of number of layers from 1 to 100----------------------------
hidden_layer =(50,)*100

ScoreResult = []
PredResult = []
RecallResult = []
LossResult = []
#print(hidden_layer)
#print(hidden_layer[0:1])

for i in range (100):
    clf = MLPClassifier(hidden_layer_sizes = hidden_layer[0:i])
    clf.fit(xtrain, ytrain)
    prediction = clf.predict(xtest)
    score = clf.score(xtest, ytest)
    precision = metrics.precision_score(ytest, prediction, average='macro')
    recall = metrics.recall_score(ytest, prediction, average='macro')
    loss0_1 = metrics.zero_one_loss(ytest, prediction)

    ScoreResult.append(score)
    PredResult.append(precision)
    RecallResult.append(recall)
    LossResult.append(loss0_1)

    print("For ", i, "hidden layer (s), The score = ", score *100, "%", ", Precision = ", precision*100, "% .." )

# Visualize results :
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(4, sharex=True, figsize=(10,10))
axarr[0].plot(range(100), ScoreResult)
axarr[0].set_title('Number of hidden layers from 1 to 99')
axarr[0].set_ylabel('Score')
axarr[1].plot(range(100), PredResult)
axarr[1].set_ylabel('Precision')
axarr[2].plot(range(100), RecallResult)
axarr[2].set_ylabel('Recall')
axarr[3].plot(range(100), LossResult)
axarr[3].set_ylabel('Zero-to-one Loss')


# ----------------------------------- 5  RNN Random models ----------------------------------
# 1 layer, max neurals
clf1 = MLPClassifier(hidden_layer_sizes=(300))
# 3 layers, random neurals
clf3 = MLPClassifier(hidden_layer_sizes=(20, 200, 50))
# 5 layers, gaussien neurals
clf5 = MLPClassifier(hidden_layer_sizes=(50, 100, 200, 100, 50))
# 7 layers, desincrese neurals :
clf7 = MLPClassifier(hidden_layer_sizes=(300, 250, 200, 150, 100, 50, 10))
# 9 layers, increase neurals :
clf9 = MLPClassifier(hidden_layer_sizes=(30, 60, 90, 120, 150, 180, 210, 240, 270))

ClassifierList = ("clf1", "clf3","clf5", "clf7", "clf9")

Score =[]
Precision = []
Recall = []
Loss = []
TimeTraining = []
TimePrediction = []

def runClfs(clf, i):

    #Training :
    startTrain =time.time()
    clf.fit(xtrain, ytrain)
    endTrain = time.time()

    #Prediction :
    startpred= time.time()
    predict = clf.predict(xtest)
    endpred = time.time()

    #Metrics :
    score = clf.score(xtest,ytest)
    precision =  metrics.precision_score(ytest, predict,  average='macro')
    recall = metrics.recall_score(ytest, predict, average ='macro')
    loss01 = metrics.zero_one_loss(ytest, predict)
    timetrain = endTrain - startTrain
    timePred = endpred - startpred

    #Saving results
    Score.append(score*100)
    Precision.append(precision*100)
    Recall.append(recall)
    Loss.append(loss01)
    TimePrediction.append(timePred)
    TimeTraining.append(timetrain)

    #Prints :
    print("For the", i," model we have, score = ", score*100, "%, precision =",precision*100, "%." )
    print("   Training's time = ", timetrain, "(s) and prediction's time = ", timePred, "(s)." )
    

# Test :
runClfs(clf1, 1)
runClfs(clf1, 2)
runClfs(clf1, 3)
runClfs(clf1, 4)
runClfs(clf1, 5)
# Visualize results :
fig, axarr = plt.subplots(6, sharex=True, figsize=(10,10))
axarr[0].scatter(range(5), Score, c='orange')
axarr[0].set_title('The five classifiers with respectively 1,3,5,7,9 hidden layers')
axarr[0].set_ylabel('Score (%)')
axarr[1].scatter(range(5), Precision, c='red')
axarr[1].set_ylabel('Precision (%)')
axarr[2].scatter(range(5), Recall, c='green')
axarr[2].set_ylabel('Recall ')
axarr[3].scatter(range(5), Loss, c='blue')
axarr[3].set_ylabel('Zero-to-one Loss')
axarr[4].scatter(range(5), TimeTraining, c='pink')
axarr[4].set_ylabel('Training Time (s)')
axarr[5].scatter(range(5), TimePrediction, c='purple')
axarr[5].set_ylabel('¨Prediction Time (s)')

plt.show()

#------------------------------------ Solvers variation (adam....)----------------------------------------
# Our tuples (<==> i layer(s) with a random number of neurals) :
t1 = (30)
t3 = (20, 200, 50)
t5 = (50, 100, 200, 100, 50)
t7 = (300, 250, 200, 150, 100, 50, 10)
t9 = (30, 60, 90, 120, 150, 180, 210, 240, 270)
TotalScore          = []
TotalPrecision      = []
TotalRecall         = []
TotalLoss           = []
TotalTrainingTime   = []
TotalPredictionTime = []

def trySolver(tuple, solv, i):
    
    #Training :
    clf = MLPClassifier(hidden_layer_sizes = tuple, solver = solv)
    start1 = time.time()
    clf.fit(xtrain, ytrain)
    end1 = time.time()

    #Prediction :
    start2 = time.time()
    prediction = clf.predict(xtest)
    end2 = time.time()

    #Metrics :
    score = clf.score(xtest, ytest)
    precision = metrics.precision_score(ytest, prediction, average = 'macro')
    recall = metrics.zero_one_loss(ytest, prediction)
    loss0_1 = metrics.zero_one_loss(ytest, prediction)

    trainingTime = end1 - start1
    predictionTime = end2 - start2

    #Saving reults :
    TotalScore.append(score)
    TotalPrecision.append(precision)
    TotalRecall.append(recall)
    TotalLoss.append(loss0_1)
    TotalTrainingTime.append(trainingTime)
    TotalPredictionTime.append(predictionTime)

    # Printing results :
    print("For the Solver :: ", solv)
    print("for the ", i, "model we have, a score = ", score * 100, "%, precision = ", precision * 100, "%, training's time = ", trainingTime, "and a prediction's time = ", predictionTime, " .")
    

# Testing :
for j in ('lbfgs', 'sgd', 'adam'):
    trySolver(t1, j, 1)
for j in ('lbfgs', 'sgd', 'adam'):
    trySolver(t1, j, 2)
for j in ('lbfgs', 'sgd', 'adam'):
    trySolver(t1, j, 3)
for j in ('lbfgs', 'sgd', 'adam'):
    trySolver(t1, j, 4)
for j in ('lbfgs', 'sgd', 'adam'):
    trySolver(t1, j, 5)

# Visualization :
fig, axarr = plt.subplots(6, sharex=True, figsize=(10,10))

axarr[0].scatter(range(15), TotalScore, c='red')
axarr[0].set_title('The five classifiers with 3 training methods : <lbfgs>, <sgd>, <adam>')
axarr[0].set_ylabel('Score (%)')
axarr[1].scatter(range(15), TotalPrecision, c='blue')
axarr[1].set_ylabel('Precision (%)')
axarr[2].scatter(range(15), TotalRecall, c='orange')
axarr[2].set_ylabel('Recall ')
axarr[3].scatter(range(15), TotalRecall, c='purple')
axarr[3].set_ylabel('Zero-to-one Loss')
axarr[4].scatter(range(15), TotalTrainingTime, c='yellow')
axarr[4].set_ylabel('Training Time (s)')
axarr[5].scatter(range(15), TotalPredictionTime)
axarr[5].set_ylabel('¨Prediction Time (s)')

plt.show()

#--------------------------- Variation of the activation function ------------------------------------------------------
TotalScore2 = []
TotalPrecision2 = []
TotalRecall2 = []
TotalLoss2 = []
TotalTrainingTime2 =[]
TotalPredictionTime2 = []
def trySomeActivationFcts(t, activation ,i) :
    #Training :
    clf = MLPClassifier(hidden_layer_sizes = t, activation = activation)
    start1 = time.time()
    clf.fit(xtrain, ytrain)
    end1= time.time()

    #Prediction :
    start2 = time.time()
    prediction = clf.predict(xtest)
    end2 = time.time()

    #Metrics :
    score = clf.score(xtest, ytest)
    precision = metrics.precision_score(ytest, prediction, average='macro')
    recall = metrics.recall_score(ytest, prediction, average='macro')
    loss0_1 = metrics.zero_one_loss(ytest, prediction)

    trainingT = end1 - start1
    predictionT = end2 - start2

    # Saving the results :
    TotalScore2.append(score)
    TotalPrecision2.append(precision)
    TotalRecall2.append(recall)
    TotalLoss2.append(loss0_1)
    TotalTrainingTime2.append(trainingT)
    TotalPredictionTime2.append(predictionT)

    #Print the results :
    print("For the activation function : ", activation)
    print("for the, ", i, " model, the score = ",score * 100, "%, precision = ", precision * 100, "%, training'time = ", trainingT, "(s) and the prediction's time is = ", predictionT, " (s).")
     
# Testing :
for j in ('identity', 'logistic', 'tanh', 'relu'):
    trySomeActivationFcts(t1, j, 1)
for j in ('identity', 'logistic', 'tanh', 'relu'):
    trySomeActivationFcts(t1, j, 2)
for j in ('identity', 'logistic', 'tanh', 'relu'):
    trySomeActivationFcts(t1, j, 3)
for j in ('identity', 'logistic', 'tanh', 'relu'):
    trySomeActivationFcts(t1, j, 4)
for j in ('identity', 'logistic', 'tanh', 'relu'):
    trySomeActivationFcts(t1, j, 5)

#Visualization :
fig, axarr = plt.subplots(6, sharex=True, figsize=(10,10))
axarr[0].scatter(range(20), TotalScore2)
axarr[0].set_title('The five classifiers with different activation functions : identity, logistic, tanh, relu')
axarr[0].set_ylabel('Score (%)')
axarr[1].scatter(range(20), TotalPrecision2)
axarr[1].set_ylabel('Precision (%)')
axarr[2].scatter(range(20), TotalRecall2)
axarr[2].set_ylabel('Recall ')
axarr[3].scatter(range(20), TotalLoss2)
axarr[3].set_ylabel('Zero-to-one Loss')
axarr[4].scatter(range(20), TotalTrainingTime2)
axarr[4].set_ylabel('Training Time(s)')
axarr[5].scatter(range(20), TotalPredictionTime2)
axarr[5].set_ylabel('¨Prediction Time(s)')
plt.show()


#-------------------------------- Variation of alpha parameter -------------------------------------------

alphas = np.logspace(-5, 3, 5) # 5 diffrents values 

FinalScore = []
FinalPrecision = []
FinalRecal = []
FinalLoss = []
FinalTimeTraining = []
FinalTimePrediction = []

def variateAlpha(numTuple, a, i):
    # Train
    clf = MLPClassifier(hidden_layer_sizes=numTuple, alpha= a)
    startTrain =time.time()
    clf.fit(xtrain, ytrain)
    endTrain = time.time()
    # Predict
    startpred= time.time()
    predict = clf.predict(xtest)
    endpred = time.time()
    # Metrics
    score = clf.score(xtest,ytest)
    precision =  metrics.precision_score(ytest, predict,  average='macro')
    recall = metrics.recall_score(ytest, predict, average ='macro')
    loss01 = metrics.zero_one_loss(ytest, predict)
    timetrain = endTrain - startTrain
    timePred = endpred - startpred
    #Append
    FinalScore.append(score*100)
    FinalPrecision.append(precision*100)
    FinalRecal.append(recall)
    FinalLoss.append(loss01)
    FinalTimeTraining.append(timePred)
    FinalTimePrediction.append(timetrain)
    #Print
    print ("ALPHA : ", a)
    print("pour le ", i,"eme modèle, score = ", score*100, "%, précision =",precision*100, "%." )
    print("    temps apprentissage : ", timetrain, "sec , temps prediction = ", timePred, "sec." )

#TESTING:
for j in alphas:
    variateAlpha(t1, j, 1)
for j in alphas:
    variateAlpha(t1, j, 2)
for j in alphas:
    variateAlpha(t1, j, 3)
for j in alphas:
    variateAlpha(t1, j, 4)
for j in alphas:
    variateAlpha(t1, j, 5)

#VISUALIZATION :
fig, axarr = plt.subplots(6, sharex=True, figsize=(10,10))
axarr[0].scatter(range(25), FinalScore)
axarr[0].set_title('The five classifiers with different alpha :  1^-5, 0.001, 0.1, 10, 1000 ')
axarr[0].set_ylabel('Score (%)')
axarr[1].scatter(range(25), FinalPrecision)
axarr[1].set_ylabel('Precision (%)')
axarr[2].scatter(range(25), FinalRecal)
axarr[2].set_ylabel('Recall ')
axarr[3].scatter(range(25), FinalLoss)
axarr[3].set_ylabel('Zero-to-one Loss')
axarr[4].scatter(range(25), FinalTimeTraining)
axarr[4].set_ylabel('Training Time in sec')
axarr[5].scatter(range(25), FinalTimePrediction)
axarr[5].set_ylabel('¨Prediction Time in sec')

plt.show()

#------------------------------ end---------------------------
