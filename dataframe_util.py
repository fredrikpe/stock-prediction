def moving_average(dataframe, window, min_periods=1):
    return dataframe.rolling(window, min_periods=min_periods).mean()


def moving_standard_dev(dataframe, window, min_periods=1):
    return dataframe.rolling(window, min_periods=min_periods).std(
        ddof=0
    )  # To avoid NaN


def df_iterator(df, start_date, end_date):
    for index, row in df.iterrows():
        date = row["Date"]
        if date < start_date:
            continue
        if date > end_date:
            break
        del row["Date"]
        yield (date, row)
