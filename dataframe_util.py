def moving_average(dataframe, window, min_periods=1):
    return dataframe.rolling(window, min_periods=min_periods).mean()


def moving_standard_dev(dataframe, window, min_periods=1):
    return dataframe.rolling(window, min_periods=min_periods).std(
        ddof=0
    )  # To avoid NaN
