import datetime
import simulation.market_simulation as ms

market = ms.load_default_market()


def test_empty_portfolio_worth():
    assert market.net_worth(ms.Portfolio(market), datetime.date(2017, 1, 1)) == 0
