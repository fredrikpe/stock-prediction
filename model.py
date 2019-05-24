import torch
import sys
import logging

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=0.2,
            bidirectional=False,
        )
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)

        self.init_hidden()

    def init_hidden(self):
        return (
            torch.zeros(
                self.num_layers, self.batch_size, self.hidden_dim, device=DEVICE
            ),
            torch.zeros(
                self.num_layers, self.batch_size, self.hidden_dim, device=DEVICE
            ),
        )

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x.view(-1, self.batch_size, self.input_dim))
        pred = self.linear.forward(lstm_out)
        return pred[-1].squeeze()

    def fit(self, training_data, num_epochs, learning_rate=1e-1):
        loss_fn = torch.nn.MSELoss(reduction="sum")
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        loss = 0
        for epoch in range(num_epochs):
            logging.info("Epoch", epoch, "Loss", loss)

            for (train, target) in training_data:
                self.zero_grad()

                pred = self.forward(train)

                loss = loss_fn(pred[-1], target[-1])

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
