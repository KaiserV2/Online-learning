import matplotlib.pyplot as plt

# ground truth
freq_arms = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
plt.plot(range(0,10), freq_arms)
plt.title("Frequence of chosen arms")
plt.show()


freq_arms = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plt.plot(range(0,10), freq_arms)
plt.title("Frequence of chosen arms")
plt.show()

freq_arms = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
plt.plot(range(0,10), freq_arms)
plt.title("Frequence of chosen arms")
plt.show()

freq_arms = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
plt.plot(range(0,10), freq_arms)
plt.title("Frequence of chosen arms")
plt.show()

# Exp3

# UCB

# AdSwitch