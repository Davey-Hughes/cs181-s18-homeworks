import matplotlib.pyplot as plt

def main():
    xs = [-3, -2, -1, 0, 1, 2, 3]
    ys = list(map(lambda x: x ** 2, xs))

    plt.scatter(xs, ys)
    plt.axhline(y=2.5)
    plt.savefig('prob1.png')


if __name__ == '__main__':
    main()
