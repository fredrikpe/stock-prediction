import random
import collections


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
            self.market.buy(self.portfolio, stock, max, date)
        else:
            self.market.sell_max(self.portfolio, stock, date)


class MovingAverage(Strategy):
    def __init__(self, name, market, num_days):
        super(MovingAverage, self).__init__(name, market)
        self.last_three = {
            stock: [0.0, collections.deque(maxlen=num_days)]
            for stock in self.market.stock_names
        }
        self.num_days = num_days

    def train(self, row, date):
        for stock, value in row.iteritems():
            self.last_three[stock][1].append(value)
            self.last_three[stock][0] = self.moving_average(stock)

    def execute(self, row, date):
        for stock, value in row.iteritems():
            self.last_three[stock][1].append(value)

            new_ma = self.moving_average(stock)
            old_ma = self.last_three[stock][0]

            amount_owned = self.portfolio.stocks[stock]
            ratio = new_ma / old_ma
            if ratio > 1.0:
                max = self.market.max_affordable_amount(self.portfolio, stock, date)
                to_buy = int(max * (ratio - 1))
                self.market.buy(self.portfolio, stock, to_buy, date)
            elif ratio < 1.0:
                to_sell = amount_owned - int(amount_owned * ratio)
                self.market.sell(self.portfolio, stock, to_sell, date)

            self.last_three[stock][0] = new_ma

    def moving_average(self, stock):
        return sum(self.last_three[stock][1]) / self.num_days
