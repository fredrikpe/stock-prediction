import pandas
import data_loader
import dataframe_util as df_util
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def _delete_some_stocks(df, stock_names):
    to_delete = [
        "BGF US Basic Value D2 {EUR}",
        "iShares North America Eq Idx (LU) A2 USD {EUR}",
        "BGF Asian Growth Leaders D2.1 {EUR}",
        "Allianz Europe Equity Growth R EUR {EUR}",
        "Allianz Europe Equity Value I EUR {EUR}",
        "iShares Europe Equity Index (LU) A2 EUR {EUR}",
        "NT All Cntry Asia exJpnCst ESG EqIdx C € {EUR}",
        "KLP AksjeNorden Indeks {EUR}",
        "C WorldWide Norden {EUR}",
        "SEB Nordenfond {EUR}",
    ]
    for name in to_delete:
        del df[name]
        stock_names.remove(name)


if __name__ == "__main__":
    df = data_loader.load_stock_data(2013, 2017)
    stock_names = data_loader.stock_names(df)
    _delete_some_stocks(df, stock_names)

    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    scaler.fit(df[stock_names])
    # df[stock_names] = scaler.transform(df[stock_names])

    fund = df[["BGF US Growth A2 {EUR}"]]

    plt.figure()

    moving_averages = pandas.DataFrame()
    for i, stock_name in enumerate(stock_names):
        moving_averages[stock_name + " MA 3"] = df_util.moving_average(
            df[stock_name], 3
        )

        # plt.subplot(5, 2, i + 1)
        plt.plot(df.Date, df[stock_name])
        # plt.title(stock_name)

    [print(x) for x in stock_names]
    plt.legend(([name for name in stock_names]))
    plt.show()
