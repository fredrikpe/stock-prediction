import datetime
import simulation.market_simulation as ms

market = ms.load_default_market()


def test_empty_portfolio_worth():
    assert (
        ms.net_worth(ms.empty_portfolio(market), market, datetime.date(2017, 1, 1)) == 0
    )
