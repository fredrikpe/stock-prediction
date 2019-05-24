import pandas

import simulation.data_loader as data_loader
from collections import namedtuple
import datetime

Portfolio = namedtuple("Portfolio", "stocks cash")


class Simulation:
    def run(self):
        pass


class Market:
    def __init__(self, dataframe, stock_names):
        self.df = dataframe
        self.stock_names = stock_names

    def stock_price(self, stock_name, amount, date):
        if amount < 0:
            raise Exception("amount can't be negative")
        price = self.df.loc[self.df.Date == date][stock_name].item() * amount
        if amount == 0:
            assert price == 0, "Error: Price of 0 stocks was not 0!"
        return price


def load_default_market():
    df = data_loader.load_stock_data(2016, 2018)
    sn = data_loader.stock_names(df)

    return Market(df, sn)


def empty_portfolio(market):
    return Portfolio(stocks={name: 0 for name in market.stock_names}, cash=0)


def net_worth(portfolio, market, date):
    return sum(
        [
            market.stock_price(stock, amount, date)
            for stock, amount in portfolio.stocks.items()
        ]
    )


def sell(portfolio, stock_name, amount, date):
    amount_owned = portfolio.stocks[stock_name]
    amount_to_sell = amount_owned if amount_owned < amount else amount

    portfolio.stocks[stock_name] -= amount_to_sell
    portfolio.cash += market.stock_price(stock_name, amount_to_sell, date)


def buy(portfolio, stock_name, amount, date):
    amount_to_buy = amount
    price = market.stock_price(stock_name, amount_to_buy, date)
    while price > portfolio.cash:
        amount_to_buy -= 1
        price = market.stock_price(stock_name, amount_to_buy, date)

    portfolio.stocks[stock_name] += amount_to_buy
    portfolio.cash -= price(amount_to_buy)


if __name__ == "__main__":
    import data_loader

    df = data_loader.load_stock_data()

    sn = data_loader.stock_names(df)

    print(stock_price(df, sn[0], 10, datetime.date(2017, 3, 3)))
