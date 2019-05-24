import random


class Strategy:
    def __init__(self, name, market):
        self.name = name
        self.market = market
        self.portfolio = market.default_portfolio()

    def train(self, row, date):
        pass

    def execute(self, row, date):
        pass


class Inactive(Strategy):
    pass


class SellImmediately(Strategy):
    def execute(self, row, date):
        for stock in self.market.stock_names:
            if self.portfolio.stocks[stock] > 0:
                self.market.sell_max(self.portfolio, stock, date)


class RandomStrategy(Strategy):
    def execute(self, row, date):
        stock = random.choice(self.market.stock_names)
        max = self.market.max_affordable_amount(self.portfolio, stock, date)
        if max > 0:
            print("amount", max, "cash", self.portfolio.cash)
            self.market.buy(self.portfolio, stock, max, date)
        else:
            self.market.sell_max(self.portfolio, stock, date)


class MovingAverage3(Strategy):
    def train(self, row, date):
        self.last_three = {}
        for stock, value in row.iteritems():
            self.last_three[stock] = value

    def execute(self, row, current_date):
        pass
