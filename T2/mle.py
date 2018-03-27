import matplotlib.pyplot as plt

data = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]

def theta_mle(data):
    return float(data.count(1)) / len(data)

def theta_map(alpha, beta, data):
    return (float(alpha) + data.count(1) - 1) / (alpha + beta + len(data) - 2)

def pos_predictive(alpha, beta, data):
    return (float(alpha) + data.count(1)) / (alpha + beta + len(data))

def main():

    mle_arr = []
    map_arr = []
    pos_predictive_arr = []
    for i in range(len(data)):
        mle_arr.append(theta_mle(data[:i + 1]))
        map_arr.append(theta_map(4, 2, data[:i + 1]))
        pos_predictive_arr.append(theta_map(4, 2, data[:i + 1]))

    X = [i + 1 for i in range(len(data))]

    mle_pts = plt.scatter(X, mle_arr, marker='x', color='r')
    map_pts = plt.scatter(X, map_arr, marker='o', color='b')
    pos_pts = plt.scatter(X, pos_predictive_arr, marker='x', color='y')

    plt.legend((mle_pts, map_pts, pos_pts), ('MLE', 'MAP', 'POS Predictive'), loc='lower right')
    # plt.show()
    plt.savefig('prob1_1.png')


if __name__ == '__main__':
    main()
