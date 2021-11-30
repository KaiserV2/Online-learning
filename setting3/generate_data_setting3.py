import numpy as np
import random

random.seed(1)
np.random.seed(1)
f1 = open("dataset2.txt",'w')

# TODO：the data needs to be in the right way (or else it cannot converge)

mu = [1100, 1100, 1100, 1100, 12, 1100, 1100, 1100, 1100, 1100] # GOOD = 4
sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

arm_length = 10
sim_length = 20000

each_change = int(sim_length / 4)

for _ in range(each_change):
    #in range(10):
    line = []
    for i in range (10):
        temp = np.random.normal(mu[i], sigma[0])
        while temp < 0:
        	temp = np.random.normal(mu[i], sigma[0])
        #print(temp, "---temp")
        line.append(temp)
    all_number = ' '.join(map(str, line))
    f1.write("%s \n" %all_number)

mu = [20, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] # GOOD = 0

for _ in range(each_change):
    #in range(10):
    line = []
    for i in range (10):
        temp = np.random.normal(mu[i], sigma[0])
        while temp < 0:
        	temp = np.random.normal(mu[i], sigma[0])
        #print(temp, "---temp")
        line.append(temp)
    all_number = ' '.join(map(str, line))
    f1.write("%s \n" %all_number)

mu = [30, 1200, 1300, 1400, 1000, 1000, 1000, 30, 1000, 1200] # GOOD = 0, 7

for _ in range(each_change):
    #in range(10):
    line = []
    for i in range (10):
        temp = np.random.normal(mu[i], sigma[0])
        while temp < 0:
        	temp = np.random.normal(mu[i], sigma[0])
        #print(temp, "---temp")
        line.append(temp)
    all_number = ' '.join(map(str, line))
    f1.write("%s \n" %all_number)

mu = [1200, 18, 1300, 1400, 1200, 1000, 1000, 15, 1000, 1200] # GOOD = 1, 7

for _ in range(each_change):
    #in range(10):
    line = []
    for i in range (10):
        temp = np.random.normal(mu[i], sigma[0])
        while temp < 0:
        	temp = np.random.normal(mu[i], sigma[0])
        #print(temp, "---temp")
        line.append(temp)
    all_number = ' '.join(map(str, line))
    f1.write("%s \n" %all_number)
