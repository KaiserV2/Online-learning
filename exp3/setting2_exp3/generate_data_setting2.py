import numpy as np
import random

random.seed(1)
np.random.seed(1)
f1 = open("dataset1.txt",'w')

mu = [2950, 3318, 4700, 4900, 3212, 4100, 4100, 745, 4100, 4500] # GOOD = 8
sigma = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

arm_length = 10
sim_length = 50000

#each_change = int(sim_length / 4)

for _ in range(sim_length):
    #in range(10):
    line = []
    for i in range (10):
        temp = np.random.normal(mu[i], sigma[0])
        while temp < 0:
            temp = np.random.normal(mu[i], sigma[0])
        #print(temp, "---temp")
        line.append(temp)
    all_number = ', '.join(map(str, line))
    f1.write("%s \n" %all_number)