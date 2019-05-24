import pandas

import datetime
from currency_converter import CurrencyConverter
from collections import namedtuple


converter = CurrencyConverter(fallback_on_missing_rate=True)


def load_stock_data(first_year, last_year):
    df = _read_stock_csvs(first_year, last_year)
    del df["Unnamed: 0"]
    df.Date = df.Date.apply(lambda d: _to_date(d))
    _convert_columns_to_eur(df)
    return df


def stock_names(dataframe):
    return [x for x in dataframe.columns.values if _get_currency(x) is not None]


def _read_stock_csvs(first_year, last_year):
    return pandas.concat(
        [
            pandas.read_csv("stock_data/target." + str(year) + ".csv")
            for year in range(first_year, last_year)
        ]
    )


def _to_date(series):
    return datetime.datetime.strptime(series, "%Y-%m-%d").date()


def _get_currency(name):
    for index, currency in enumerate(["{USD}", "{NOK}", "{SEK}", "{EUR}"]):
        if name.endswith(currency):
            return currency[1:-1]
    return None


def _convert_to_eur(x, currency, column_name):
    return converter.convert(x[column_name], currency, "EUR", date=x["Date"])


def _convert_columns_to_eur(dataframe):
    for column_name in dataframe.columns.values:
        currency = _get_currency(column_name)
        if currency is not None and currency != "EUR":
            print("Converting column from", currency, "to EUR")
            new_name = column_name[:-5] + "{EUR}"
            dataframe[new_name] = dataframe.apply(
                lambda x: _convert_to_eur(x, currency, column_name), axis=1
            )
            del dataframe[column_name]


if __name__ == "__main__":
    df = get_stock_dataframe()

    print(df.loc[df.Date == datetime.date(2017, 2, 2)])
    print(stock_names(df))
