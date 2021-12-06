import os
import numpy as np
import random
from math import *
from matplotlib import pyplot as plt

# change these parameters 
n = 10 # the number of bandits
T = 20000 # the number of rounds
Segment = 4
alpha = 50.0

lowerConfidenceBound = np.zeros(n)
hatMiu = np.zeros(n)
s = np.zeros(n) # the number each bandit is pulled
total_loss = 0.0
averageCumulativeLoss = []
prediction = []
confidenceBound = np.zeros(n)
truePrediction = []

def read_in_data ():
    tempLatency = []
    rs = os.path.exists('dataset1.txt')
    if rs == True:
        print("Successfully read in dataset \n")
        file_handler = open('dataset1.txt',mode='r')  
        contents = file_handler.readlines()
        for name in contents:
            name = name.strip('\n')
            list_1 = name.split(', ')
            tempLatency.append(list_1)
    latency = np.array(tempLatency)
    return latency

latency = read_in_data()

def true_prediction():
    for i in range(T):
        #print(latency[i])
        index = np.argmin(latency[i][0:10].astype(np.float))
        #print(index)
        truePrediction.append(index)

true_prediction()

def print_confidence_bar (y, y_err, time):
    # print(y_err)
    x1_ = np.arange(n)
    plt.rcParams['font.size'] = 22
    plt.plot(x1_, y, color = 'b', label = 'Estimated mean value')
    plt.errorbar(x1_, y, yerr = y_err, color = 'r', fmt = 'o', label = 'Confidence bar')
    plt.title( "Confidence bar at %d-th round" %time)
    plt.show()

for t in range(T):
    # update the lower confidence bound
    for j in range(n):
        confidenceBound[j] = sqrt(2 * alpha * (t + 1) / (s[j] + 1) )
        #confidenceBound[j] = sqrt(2 * alpha * log(t + 1) / (s[j] + 1) )
        lowerConfidenceBound[j] = hatMiu[j] - confidenceBound[j] # can change log into loglog

    # plot the confidence bound    
    if ((t+1) % (T / Segment) == 0):
        print_confidence_bar(hatMiu, confidenceBound, t + 1)
        
    # choose the one with minimal lower confidence bound
    min_index = np.argmin(lowerConfidenceBound)
    prediction.append(min_index)

    # receive loss
    loss = float(latency[t][min_index]) / 4
    total_loss += loss
    averageCumulativeLoss.append(total_loss / (t + 1))
    s[min_index] += 1
    hatMiu[min_index] = (hatMiu[min_index] * (s[min_index] - 1) + loss) / s[min_index]

# print the average cumulative loss of T rounds
def print_avgcum ():
    x1_ = np.arange(T)
    plt.plot(x1_, averageCumulativeLoss, color = 'b', label = 'UCB: t, alpha = %d' %alpha)
    plt.title( "Average Cumulative Loss")
    plt.legend()
    plt.rcParams['font.size'] = 22
    plt.show()

# print the prediction distribution between a period of time
def print_prediction_distribution (begin, end):
    counter = np.zeros(n)
    trueCounter = np.zeros(n)
    for i in range(end - begin):
        counter[prediction[begin + i]] += 1
        trueCounter[truePrediction[begin + i]] += 1
    x1_ = np.arange(n)
    counter /= (T / Segment)
    trueCounter /= (T / Segment)
    plt.plot(x1_, counter, color = 'b', label = 'UCB: t, alpha = %d' %alpha)
    plt.plot(x1_, trueCounter, color = 'r', label = 'Ground Truth', linestyle = '--')
    #plt.title( "Frequence of chosen arms t = %d ... %d" %(begin+1, end))
    plt.title( "Frequence of chosen arms")
    plt.legend()
    plt.rcParams['font.size'] = 22
    plt.show()


print_avgcum()
print_prediction_distribution(0, int(T / Segment))
print_prediction_distribution(int(1 * T / Segment), int(2 * T / Segment))
print_prediction_distribution(int(2 * T / Segment), int(3 * T / Segment))
print_prediction_distribution(int(3 * T / Segment), int(4 * T / Segment))