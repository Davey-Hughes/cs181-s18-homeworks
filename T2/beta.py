from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

X = [0, 4, 8, 12, 16]

x = np.arange(0.01, 1, 0.01)

i_0, = plt.plot(x, beta.pdf(x, 4, 2))
i_4, = plt.plot(x, beta.pdf(x, 6, 4))
i_8, = plt.plot(x, beta.pdf(x, 6, 8))
i_12, = plt.plot(x, beta.pdf(x, 9, 9))
i_16, = plt.plot(x, beta.pdf(x, 11, 11))
plt.legend((i_0, i_4, i_8, i_12, i_16), ('0', '4', '8', '12', '16'))
plt.savefig('prob1.2.png')
