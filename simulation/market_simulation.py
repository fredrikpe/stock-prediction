import pandas

import dataframe_util as util
import simulation.data_loader as data_loader
from collections import namedtuple
import datetime


class Portfolio:
    def __init__(self, market):
        self.stocks = stocks = {name: 0 for name in market.stock_names}
        self.cash = 0


class Market:
    def __init__(self, dataframe, stock_names):
        self.df = dataframe
        self.stock_names = stock_names

    def run(self, strategies, start_date, training_end_date, end_date):
        for date, row in util.df_iterator(self.df, start_date, training_end_date):
            for strategy in strategies:
                strategy.train(row, date)

        result = {strategy: [] for strategy in strategies}
        for date, row in util.df_iterator(self.df, training_end_date, end_date):
            for strategy in strategies:
                strategy.execute(row, date)
                result[strategy].append(self.net_worth(strategy.portfolio, date))

        return result

    def stock_price(self, stock_name, amount, date):
        assert amount >= 0, "Error: amount can't be negative"
        price = self.df.loc[self.df.Date == date][stock_name].item() * amount
        assert amount > 0 or price == 0, "Error: Price of 0 stocks was not 0!"
        return price

    def sell(self, portfolio, stock_name, amount, date):
        amount_owned = portfolio.stocks[stock_name]
        amount_to_sell = amount_owned if amount_owned < amount else amount

        portfolio.stocks[stock_name] -= amount_to_sell
        portfolio.cash += self.stock_price(stock_name, amount_to_sell, date)

    def buy(self, portfolio, stock_name, amount, date):
        amount_to_buy = amount
        price = self.stock_price(stock_name, amount_to_buy, date)
        while price > portfolio.cash:
            amount_to_buy -= 1
            price = self.stock_price(stock_name, amount_to_buy, date)

        portfolio.stocks[stock_name] += amount_to_buy
        portfolio.cash -= price
        return portfolio

    def sell_max(self, portfolio, stock_name, date):
        amount_owned = portfolio.stocks[stock_name]
        self.sell(portfolio, stock_name, amount_owned, date)

    def max_affordable_amount(self, portfolio, stock_name, date):
        price_for_one = self.stock_price(stock_name, 1, date)
        return portfolio.cash // price_for_one

    def net_worth(self, portfolio, date):
        return (
            sum(
                [
                    self.stock_price(stock, amount, date)
                    for stock, amount in portfolio.stocks.items()
                ]
            )
            + portfolio.cash
        )

    def default_portfolio(self):
        portfolio = Portfolio(self)
        for stock, value in portfolio.stocks.items():
            portfolio.stocks[stock] = 100
        portfolio.cash = 100
        return portfolio



def load_default_market():
    df = data_loader.load_stock_data(2016, 2018)
    sn = data_loader.stock_names(df)

    return Market(df, sn)



if __name__ == "__main__":
    import data_loader

    df = data_loader.load_stock_data()

    sn = data_loader.stock_names(df)

    print(stock_price(df, sn[0], 10, datetime.date(2017, 3, 3)))
