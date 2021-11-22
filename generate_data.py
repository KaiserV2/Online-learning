import numpy as np
import random

f1 = open("dataset1.txt",'w')

f2 = open("dataset2.txt",'w')

mu = [8.0, 9.0, 10.0, 11.0, 12.0, 8.0, 9.0, 10.0, 11.0, 12.0]
sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]

simulation_length = 1000

for i in range(10):
    s = np.random.normal(mu[i], sigma[i], simulation_length)
    f1.write("%s \n" %s)

simulation_length //= 10

for i in range(10):
    s = []
    for j in range(10):
        # There are 10 changes in mean
        mu[i] += random.random() * 10 - 5
        print(mu[i])
        temp = np.random.normal(mu[i], sigma[i], simulation_length)
        temp = [x for x in temp]
        s += temp
    f2.write("%s \n" %s)

