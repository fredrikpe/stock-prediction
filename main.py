#!/usr/bin/env python3

import sys
import torch
import pandas
import numpy as np
import talib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import model
import dataframe_util as df_util


def make_training_data():
    training_data = []
    for i in range(0, len(csv_data) - SEQ_SIZE - 1, SKIP_SIZE):
        train = torch.tensor(csv_data[i : i + SEQ_SIZE].values, device=DEVICE).view(
            SEQ_SIZE, BATCH_SIZE, INPUT_DIM
        )
        # .float()
        target = torch.tensor(
            csv_data[i + SEQ_SIZE : i + SEQ_SIZE + 1].values, device=DEVICE
        ).view(-1)
        # .float()

        training_data.append((train, target))
    return training_data


def train(model, training_data, loss_fn, optimiser, batch_size=1):
    for (
        train,
        target,
    ) in training_data:  # i in range(0, len(training_data) - batch_size, batch_size):
        # train = torch.tensor([x[0] for x in training_data[i:batch_size]])
        # target = torch.tensor([x[0] for x in training_data[i:batch_size]])
        model.zero_grad()

        pred = model.forward(train)

        loss = loss_fn(pred[-1], target[-1])
        # print("Target", target, "Loss", loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return loss.item()


def predict():
    predictions = []
    with torch.no_grad():
        for (train, target) in training_data:
            pred = model.forward(train).data[-1].item()
            # print(pred)
            predictions.append(pred)
    return predictions


def plot(real, predictions):
    plt.plot(real)
    plt.plot(predictions)
    plt.legend((list(csv_data.columns.values)[-1], "predictions"))
    plt.show()


if __name__ == "__main__":
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_tensor_type("torch.DoubleTensor")

    print(DEVICE)

    torch.manual_seed(1)

    # Germany: Stock Market, Germany: Currency, Germany: Interest Rate, BGF US Growth A2 {USD}
    csv_data = pandas.read_csv("data.source.csv").iloc[:, 1:-1]
    fund = csv_data[["BGF US Growth A2 {USD}"]]

    csv_data.insert(0, "BGF MA 3", df_util.moving_average(fund, 3))
    csv_data.insert(0, "BGF MA 10", df_util.moving_average(fund, 10))
    csv_data.insert(0, "BGF MA 30", df_util.moving_average(fund, 30))
    csv_data.insert(0, "BGF Standard dev", df_util.moving_standard_dev(fund, 5))
    csv_data.insert(0, "RSI", talib.RSI(fund.values.squeeze(), timeperiod=9))

    csv_data = csv_data.fillna(0)

    # del csv_data['BGF US Growth A2 {USD}']

    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
    scaler.fit(csv_data)
    csv_data.iloc[:, :] = scaler.transform(csv_data)[:, :]

    INPUT_DIM = len(csv_data.columns)
    OUTPUT_DIM = INPUT_DIM
    HIDDEN_DIM = 15
    BATCH_SIZE = 1
    NUM_LAYERS = 1
    NUM_EPOCHS = 1
    SEQ_SIZE = 60
    SKIP_SIZE = 1

    model = model.LSTMModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        num_layers=NUM_LAYERS,
    )
    model.to(device=DEVICE)

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-1)

    training_data = make_training_data()
    print(training_data[0][0].shape[0])

    real = csv_data.iloc[:, -1]
    """
    loss = 0
    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch, "Loss", loss)
        loss = train(model, training_data, loss_fn, optimiser, batch_size=BATCH_SIZE)

        predictions = predict()
        plot(real, predictions)
    """
    plot(real, csv_data[["BGF MA 3"]])
