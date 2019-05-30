#!/usr/bin/env python3

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import datetime
import simulation.plotting as plotting
import simulation.strategy as strategy
import simulation.market_simulation as market_simulation

from datetime import date, timedelta


def dates_between(start, end):
    delta = end - start
    return [start + timedelta(days=i) for i in range(delta.days + 1)]


def inactive_strategy(*args, **kwarg):
    pass


def moving_average():
    pass


if __name__ == "__main__":
    market = market_simulation.load_market(2015, 2018)

    start_date = datetime.date(2015, 1, 1)
    training_end = datetime.date(2015, 2, 1)
    end_date = datetime.date(2017, 11, 2)

    strategies = [
        strategy.Inactive("inactive", market),
        strategy.RandomStrategy("random", market),
        strategy.MovingAverage("ma1", market, 1),
        strategy.MovingAverage("ma5", market, 5),
        strategy.MovingAverage("ma3", market, 3),
        # strategy.MovingAverage("ma7", market, 7),
    ]

    result = market.run(strategies, start_date, training_end, end_date)

    dates = dates_between(training_end, end_date)

    for strategy in strategies:
        plt.plot(dates, result[strategy])
    plt.legend(tuple([strategy.name for strategy in strategies]))
    plt.show()
