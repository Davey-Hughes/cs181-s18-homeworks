#####################
# CS 181, Spring 2016
# Homework 1, Problem 3
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
# plt.figure(1)
# plt.plot(years, republican_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.figure(2)
# plt.plot(years, sunspot_counts, 'o')
# plt.xlabel("Year")
# plt.ylabel("Number of Sunspots")
# plt.figure(3)
# plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
# plt.xlabel("Number of Sunspots")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions


def loss(X, Y, w):
    return np.sum(np.power(Y - np.dot(X, w), 2))

def basis_a(x):
    X = [np.ones(x.shape).T]

    for i in range(1, 6):
        X.append(x ** i)

    return np.vstack(X).T

def basis_b(x):
    X = [np.ones(x.shape).T]

    for i in range(1960, 2015, 5):
        X.append(np.exp(float(-1) / 25 * np.power(x - i, 2)))

    return np.vstack(X).T

def basis_cd(x, end):
    X = [np.ones(x.shape).T]

    for i in range(1, end):
        X.append(np.cos(x / float(i)))

    return np.vstack(X).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def moore_penrose(X, Y):
    return np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)

def part_a(Y):
    X = basis_a(years)
    inv = moore_penrose(X, Y)
    grid_X = basis_a(grid_years)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel('Year')
    plt.ylabel('Number of Republicans in Congress')
    plt.savefig('part_a_years.png')
    plt.show()

    print("Loss part a", loss(X, Y, inv))


def part_b(Y):
    X = basis_b(years)
    inv = moore_penrose(X, Y)
    grid_X = basis_b(grid_years)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel('Year')
    plt.ylabel('Number of Republicans in Congress')
    plt.savefig('part_b_years.png')
    plt.show()

    print("Loss part b", loss(X, Y, inv))


def part_c(Y):
    X = basis_cd(years, 6)
    inv = moore_penrose(X, Y)
    grid_X = basis_cd(grid_years, 6)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel('Year')
    plt.ylabel('Number of Republicans in Congress')
    plt.savefig('part_c_years.png')
    plt.show()

    print("Loss part c", loss(X, Y, inv))


def part_d(Y):
    X = basis_cd(years, 26)
    inv = moore_penrose(X, Y)
    grid_X = basis_cd(grid_years, 26)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel('Year')
    plt.ylabel('Number of Republicans in Congress')
    plt.savefig('part_d_years.png')
    plt.show()

    print("Loss part d", loss(X, Y, inv))

grid_sunspots = np.linspace(min(sunspot_counts), max(sunspot_counts), 200)

def part_a_sunspot(Y):
    X = basis_a(sunspot_counts[years < last_year])
    inv = moore_penrose(X, Y)
    grid_X = basis_a(grid_sunspots)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(sunspot_counts[years < last_year], Y,'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig('part_a_sunspots.png')
    plt.show()

    print("Loss part a_sunspots", loss(X, Y, inv))

def part_c_sunspot(Y):
    X = basis_cd(sunspot_counts[years < last_year], 6)
    inv = moore_penrose(X, Y)
    grid_X = basis_cd(grid_sunspots, 6)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(sunspot_counts[years < last_year], Y,'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig('part_c_sunspots.png')
    plt.show()

    print("Loss part c_sunspots", loss(X, Y, inv))

def part_d_sunspot(Y):
    X = basis_cd(sunspot_counts[years < last_year], 26)
    inv = moore_penrose(X, Y)
    grid_X = basis_cd(grid_sunspots, 26)
    grid_Yhat = np.dot(grid_X, inv)

    plt.plot(sunspot_counts[years < last_year], Y,'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig('part_d_sunspots.png')
    plt.show()

    print("Loss part d_sunspots", loss(X, Y, inv))


# grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
# grid_Yhat  = np.dot(grid_X.T, w)

# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()

part_a(Y)
part_b(Y)
part_c(Y)
part_d(Y)

part_a_sunspot(republican_counts[years < last_year])
part_c_sunspot(republican_counts[years < last_year])
part_d_sunspot(republican_counts[years < last_year])
