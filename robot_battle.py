import data_loader


if __name__ == "__main__":
    df = data_loader.load_stock_data(2013, 2017)
    stock_names = data_loader.stock_names(df)

    pf = Portfolio(stock_names)
