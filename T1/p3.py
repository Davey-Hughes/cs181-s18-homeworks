import matplotlib.pyplot as plt
import numpy as np

def poly(x, a, K):
    return np.array([a[i] * (x ** i) for i in range(K + 1)]).sum(0)

def gen_data(K, N):
    a = np.random.uniform(-1, 1, K + 1)
    xi = np.random.uniform(-5, 5, N)

    sigma = ((poly(xi, a, K).max() - poly(xi, a, K).min()) / float(10))

    yi = poly(xi, a, K) + np.random.normal(0, sigma, N)

    return xi, yi, sigma

def min_chi2(K, sigma, x, y):
    [_, chi2, _, _, _] = np.polyfit(x, y, K, full=True)

    if K > len(x) or len(chi2) == 0:
        return float(0)

    return chi2[0] / (sigma ** 2)

def partc(K, N, num_trials):
    best_k = np.zeros(num_trials)

    for i in range(num_trials):
        xi, yi, sigma = gen_data(K, N)
        chi2 = np.array([min_chi2(R, sigma, xi, yi) for R in range(2, 30)])
        BIC = np.array([.5 * N * np.log(2 * np.pi * sigma) - N * np.log(float(1) / 10) + .5 * (chi2[k - 2] + (k + 1) * np.log(N)) for k in range(2, 30)])
        best_k[i] = np.argmin(BIC) + 2

    return best_k.mean(), best_k.std()

def partd(K, num_trials):
    n = np.round(3 * np.logspace(0, 4, 40)).astype(np.int)
    best_mean = np.zeros(40)
    best_std = np.zeros(40)

    for i in range(40):
        print(i)
        mean, std = partc(K, n[i], num_trials)
        print(mean, std)
        best_mean[i] = mean
        best_std[i] = std

    return n, best_mean, best_std

def main():
    # xi, yi, sigma = gen_data(10, 20)

    # plt.plot(range(2, 19), np.array([min_chi2(R, sigma, xi, yi) for R in range(2, 19)]))
    # plt.savefig('p3_part_b.png')
    # plt.show()

    # mean, std = partc(10, 20, 500)

    n, mean, std = partd(20, 50)

    axes = plt.axes()
    axes.set_xscale('log')
    plt.errorbar(n, mean, yerr=std)
    plt.xlabel('num data points (log scale)')
    plt.ylabel('optimal K')
    plt.savefig('q3_part_d.png')
    plt.show()

if __name__ == '__main__':
    main()
