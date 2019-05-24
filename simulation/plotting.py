import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot(values, dates):
    plt.plot(dates, values)


def show():
    plt.show()
