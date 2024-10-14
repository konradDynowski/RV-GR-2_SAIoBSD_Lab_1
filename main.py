import math

import numpy as np
from matplotlib import pyplot as plt

temperatures = [22, 24, 19, 21, 23, 25, 20, 18, 22, 24, 26, 27]

def plot(xs, ys, filename='fig.png'):
    xaxis = range(max(len(xs), len(ys)))
    plt.plot(xaxis, xs, marker='o', label='Input', color='b')  # First series
    plt.plot(xaxis, ys, marker='x', label='Transformed', color='r')  # Second series

    # Add title and labels
    plt.title(filename)
    plt.xlabel('X Axis')
    plt.ylabel('Values')

    # Show legend
    plt.legend()

    # Display the plot with gridlines
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    plot2axises(xs, ys, filename)

def plot2axises(xs, ys, filename='fig.png'):
    xaxis = range(max(len(xs), len(ys)))
    fig, ax1 = plt.subplots()

    ax1.plot(xaxis, xs, "b-", label="temperature input")
    ax1.set_ylabel("Input")

    ax2 = ax1.twinx()
    ax2.plot(xaxis, ys, "r-", label="transformed_data")

    plt.title("2fig_" + filename)
    plt.grid(True)
    plt.savefig("2fig_" + filename)
    plt.close()

# Normalization
# blue - incoming, red - transitioned
def normalize_entity(x, x_min, x_max):
    assert x_max != x_min
    return (x - x_min) / (x_max - x_min)

def normalize_set(set_x):
    max_x = np.max(set_x)
    min_x = np.min(set_x)

    new_set_x = []
    for x in set_x:
        new_set_x.append(normalize_entity(x, min_x, max_x))
    return new_set_x

normalized_temperatures = normalize_set(temperatures)
plot(temperatures, normalized_temperatures, 'normalized.png')

# standarization
def standardize(xi, xavg, delt):
    return (xi - xavg) / delt

def standardzie_set(set_x):
    x_avg = np.average(set_x)
    std_dev = np.std(set_x)

    new_set_x = []
    for x in set_x:
        new_set_x.append(standardize(x, x_avg, std_dev))
    return new_set_x

standardized_tempartures = standardzie_set(temperatures)

#print(standardized_tempartures)
plot(temperatures, standardized_tempartures, 'standardized.png')

def norm_sigm(x):
    return 1 / (1 + math.e ** (-x))

def norm_sigm_set(set_x):
    new_set_x = []
    for x in set_x:
        new_set_x.append(norm_sigm(x))
    return new_set_x

std_sigm_temp = norm_sigm_set(temperatures)
plot(temperatures, std_sigm_temp, 'std_sigm.png')


def norm_sigm2(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def norm_sigm2_set(set_x):
    new_set_x = []
    for x in set_x:
        new_set_x.append(norm_sigm2(x))
    return new_set_x

std_sigm2_temp = norm_sigm2_set(temperatures)
plot(temperatures, std_sigm2_temp, 'std_sigm2.png')

def smoothing_avg(position, set_x, m_window):
    m_2 = math.floor(m_window / 2)
    start = max(0, position - m_2)
    stop = min(len(set_x), position + m_2 + 1)
    print("start stop" + str(start) + " " + str(stop))
    to_floor = set_x[start:stop]
    print(to_floor)
    return (1 / len(to_floor)) * (np.sum(to_floor))

def smoothing_avg_set(set_x, m_window=3):
    new_set_x = []
    for id, x in enumerate(set_x):
        new_set_x.append(smoothing_avg(id, set_x, m_window))
    return new_set_x

smoothed_avg_temperatures = smoothing_avg_set(temperatures)
plot(temperatures, smoothed_avg_temperatures, "smoothed_avg.png")

def smooth_median(position, set_x, m_window):
    m_2 = math.floor(m_window / 2)
    start = max(0, position - m_2)
    stop = min(len(set_x), position + m_2 + 1)
    print("start stop" + str(start) + " " + str(stop))
    to_floor = set_x[start:stop]
    print(to_floor)
    return np.median(to_floor)

def smooth_median_set(set_x, m_window=3):
    new_set_x = []
    for id, x in enumerate(set_x):
        new_set_x.append(smooth_median(id, set_x, m_window))
    return new_set_x

smooth_med_temps = smooth_median_set(temperatures)
plot(temperatures, smooth_med_temps, "sort_median.png")


def smooth_cut_avg(position, set_x, m_window, cutoff):
    m_2 = math.floor(m_window / 2)
    assert cutoff <= m_2
    start = max(0, position - m_2)
    stop = min(len(set_x), position + m_2 + 1)
    print("start stop" + str(start) + " " + str(stop))
    to_floor = set_x[start:stop]
    sorted_to_floor = np.sort(to_floor)
    return np.mean(sorted_to_floor[cutoff:-cutoff])

def smooth_cut_avg_set(set_x):
    new_set_x = []
    for id, x in enumerate(set_x):
        new_set_x.append(smooth_cut_avg(id, set_x, 5, 1))
    return new_set_x

smoothed_avg_temperatures = smooth_cut_avg_set(temperatures)
plot(temperatures, smoothed_avg_temperatures, "smoothed_avg_cut.png")