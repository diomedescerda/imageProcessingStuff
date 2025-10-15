import matplotlib.pyplot as plt
from histogram import grayHistogram
import numpy as np

hist = grayHistogram()

J_theta = None

for i in range(hist.flatten().shape[0]):
    p_0 = hist[:i].sum() / hist.sum()
    p_1 = hist[i:].sum() / hist.sum()

    if (len(hist[:i]) == 0) or (len(hist[i:]) == 0):
        continue

    std_0 = np.std(hist[:i])
    std_1 = np.std(hist[i:])

    J = p_0 * (std_0 ** 2) + p_1 * (std_1 ** 2)

    if J_theta is None or J > J_theta:
        J_theta = J
        theta = i

plt.plot(hist)
plt.axvline(x=theta, color='r', linestyle='--', label=f'Theta = {theta}')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()