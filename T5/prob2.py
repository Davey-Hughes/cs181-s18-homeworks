import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

# constants
sigma_e2 = .05 * .05
sigma_g2 = 1.0
mu_p = 5
sigma_p2 = 1

def update(x_curr, mu_prev, sigma_prev2):
    # kalman algo filtering update
    K = (sigma_e2 + sigma_prev2) / (sigma_e2 + sigma_prev2 + sigma_g2)
    mu_next = mu_prev + K * (x_curr - mu_prev)
    sigma_next2 = K * sigma_g2
    return (mu_next, sigma_next2)

def main():
    data = pd.read_csv('kf-data.csv', index_col=0)
    xs, zs = data.x.values, data.z.values
    n = len(xs)
    preds = []
    mu_hat, sigma_hat2 = mu_p, sigma_p2
    for x in xs:
        mu_hat, sigma_hat2 = update(x, mu_hat, sigma_hat2)
        preds.append((mu_hat, sigma_hat2))

    plt.figure(figsize=(10,6))
    plt.errorbar(range(n), [p[0] for p in preds], yerr=[2*p[1] for p in preds], \
                 c='r', label='Kalman Predictions', fmt='o')
    plt.scatter(range(n), zs, c='b', label='True z')
    plt.xlabel('Timepoint')
    plt.legend()
    plt.savefig('kalman1.png')

    new_xs = deepcopy(xs)
    new_xs[11] = 10.2
    preds = []
    mu_hat, sigma_hat2 = mu_p, sigma_p2
    for x in new_xs:
        mu_hat, sigma_hat2 = update(x, mu_hat, sigma_hat2)
        preds.append((mu_hat, sigma_hat2))
    plt.figure(figsize=(10,6))
    plt.errorbar(range(n), [p[0] for p in preds], yerr=[2 * p[1] for p in preds], \
                 c='r', label='Kalman Predictions', fmt='o')
    plt.scatter(range(n), zs, c='b', label='True z')
    plt.xlabel('Timepoint')
    plt.legend()
    plt.savefig('kalman2.png')

if __name__ == '__main__':
    main()
