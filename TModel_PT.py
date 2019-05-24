#######
import TModel
import TTradeEconomics
import collections
import torch
import numpy
import datetime
import vbLib
import sys
import os
import time
from _operator import index
from builtins import str

sys.path.append(os.getenv("HOME") + "/VBlibs/Langs/python")
#######
#######
#######
###
# https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
# Here we define our model as a class


class LSTM(torch.nn.Module):
    #
    #    def __init__(self, input_dim , hidden_dim , batch_size , output_dim , num_layers ):
    # self.input_dim = input_dim
    # self.hidden_dim = hidden_dim
    # self.num_layers = num_layers
    def __init__(self, features, time_steps, cells, layers, output_dim, dropout):
        super(LSTM, self).__init__()
        #
        # VB:
        self.output_dim = output_dim
        self.time_steps = time_steps
        #
        self.input_dim = features
        #
        self.hidden_dim = cells
        #
        self.num_layers = layers

        self.learning_rate = 1e-1

        if True:
            # Define the LSTM layer
            self.lstm = torch.nn.LSTM(
                self.input_dim,
                self.hidden_dim,
                self.num_layers,
                dropout=dropout,
                bidirectional=False,
            )

            # Define the output layer
            self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
        else:  # Sequential:
            pModels = []
            if False:
                pModels.append(
                    (
                        "lstm_0",
                        torch.nn.LSTM(
                            self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=dropout,
                            bidirectional=False,
                        ),
                    )
                )
                self.Seq = torch.nn.Sequential(collections.OrderedDict(pModels))
                self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
            else:
                pModels.append(
                    torch.nn.LSTM(
                        self.input_dim,
                        self.hidden_dim,
                        1,
                        dropout=dropout,
                        bidirectional=False,
                    )
                )
                pModels.append(
                    torch.nn.LSTM(
                        self.hidden_dim,
                        self.hidden_dim,
                        1,
                        dropout=dropout,
                        bidirectional=False,
                    )
                )
                self.Seq = torch.nn.Sequential(*pModels)
                self.linear = torch.nn.Linear(self.hidden_dim, output_dim)

        params = list(self.parameters())
        print(len(params))
        print(params[0].size())  # conv1's .weight

    ### smart exit: ###
    #        pc = "['__init__:',len(params),params[0].size()]"; vbLib.exitHERE(say=eval(pc),title=pc)
    ### smart exit. ###

    #
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        ### smart exit: ###
        # TODO:?        pc = "['init_hidden:',self.num_layers]"; vbLib.exitHERE(say=eval(pc),title=pc)
        ### smart exit. ###
        return (
            torch.zeros(self.num_layers, self.time_steps, self.hidden_dim),
            torch.zeros(self.num_layers, self.time_steps, self.hidden_dim),
        )

    #
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        # if not isinstance( input , torch.Tensor ) : input = torch.Tensor( input )
        if not isinstance(input, torch.autograd.Variable):
            raise Exception(
                "Bad input type: %s (must be torch.autograd.Variable)." % (type(input))
            )

        # x_train = input.view(len(input), self.batch_size, -1)
        nTimes = input.shape[0]
        ### x_train = input.view( -1 , nTimes , self.input_dim )
        x_train = input

        # Inputs: input, (h_0, c_0)
        # input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
        #
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        #
        # c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        #
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # output, (hn, cn) = rnn(input, (h0, c0))
        # Outputs: output, (h_n, c_n)
        # output of shape (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
        # For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively. Similarly, the directions can be separated in the packed case.
        #
        # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.
        # Like output, the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size) and similarly for c_n.
        #
        # c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len.
        #
        # lstm_out , self.hidden = self.lstm( x_train )
        if hasattr(self, "Seq"):
            seq_out, self.hidden = self.Seq(x_train)
            #            pc = "['forward:', nTimes , seq_out.shape , self.hidden[0].shape ]"; vbLib.exitHERE(say=eval(pc),title=pc)
            y_pred = self.linear(seq_out)
            #            pc = "['forward:', nTimes , y_pred.shape ]"; vbLib.exitHERE(say=eval(pc),title=pc)
            retVal = y_pred[:, -1, :].view(nTimes, self.output_dim)
        #            pc = "['forward:', nTimes , retVal.shape ]"; vbLib.exitHERE(say=eval(pc),title=pc)
        else:
            lstm_out, _ = self.lstm(x_train)

            # Only take the output from the final timetep
            # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
            ### y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
            y_pred = self.linear(lstm_out)
            ### retVal = y_pred.view( nTimes , self.output_dim )
            retVal = y_pred[:, -1, :].view(nTimes, self.output_dim)

        return retVal

    #
    #############
    #############
    #############
    def fit(
        self, train_X, train_y, epochs, batch_size, validation_data, verbose, shuffle
    ):
        retVal = THistory()
        model = self
        loss_fn = torch.nn.MSELoss()  # size_average=False)
        loss_fn_val = torch.nn.MSELoss()  # size_average=False)

        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        #####################
        # Train model
        #####################

        hist = numpy.zeros(epochs)
        hist_val = numpy.zeros(epochs)
        nBatches = int(train_X.shape[0] / batch_size)
        # nBatches = min( nBatches , 3 )
        prepend = train_X.shape[0] % batch_size
        if prepend > 0:
            nBatches += 1

        # Initialise hidden state
        model.hidden = model.init_hidden()

        for t in range(epochs):
            # Clear stored gradient
            model.zero_grad()

            # Forward pass
            if verbose > 1:
                print()
            fig = vbLib.figure(dpi=80)
            # iBatch:
            for iBatch in range(nBatches):

                ### Train: ###
                if prepend == 0:
                    iBatchSize = batch_size
                    iBatch_0 = iBatch * batch_size
                elif iBatch == 0:
                    iBatchSize = prepend
                    iBatch_0 = 0
                else:
                    iBatchSize = batch_size
                    iBatch_0 = prepend + (iBatch - 1) * batch_size

                # Fill:
                y_pred = model(train_X[iBatch_0 : iBatch_0 + iBatchSize])
                y_train = torch.Tensor(train_y[iBatch_0 : iBatch_0 + iBatchSize]).view(
                    iBatchSize, self.output_dim
                )

                if False:
                    vbLib.plt.subplot(2, 1, 1)
                    vbLib.plt.cla()
                    vbLib.plt.title("train %de%db" % (t, iBatch))
                    vbLib.plt.plot(y_pred.detach().numpy())
                    vbLib.plt.plot(y_train.detach().numpy())
                    #
                    vbLib.plt.subplot(2, 1, 2)
                    vbLib.plt.cla()
                    vbLib.plt.title("state %de%db" % (t, iBatch))
                    _z = self.hidden[0].detach().view(self.num_layers, -1).numpy()
                    print(_z.shape)
                    vbLib.plt.pcolormesh(_z)  # ; vbLib.plt.colorbar()
                    vbLib.plt.xlabel("cell x timeSteps")
                    vbLib.plt.ylabel("layer")
                    #
                    vbLib.plt.pause(1e-2)
                    # vbLib.ask()

                loss = loss_fn(y_pred, y_train)
                ### Train. ###
                hist[t] = loss.item()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            ### Validate: ###
            y_pred = model(validation_data[0])

            y_train = torch.Tensor(validation_data[1])

            loss_val = loss_fn_val(y_pred, y_train)

            hist_val[t] = loss_val.item()
            ### Validate. ###

        print("\n")
        #
        retVal.history = {"loss": list(hist), "val_loss": list(hist_val)}
        #
        #
        ### smart exit: ###
        #        pc = "['LSTM.fit():',train_X, train_y, epochs, batch_size, validation_data, verbose, shuffle]"; vbLib.exitHERE(say=eval(pc),title=pc)
        ### smart exit. ###
        return retVal

    #
    #

    def predict(self, scaled_framed_X):
        model = self
        y_pred = model(torch.autograd.Variable(torch.Tensor(scaled_framed_X)))
        # y_pred = numpy.array( y_pred )
        y_pred = y_pred.detach().numpy()
        return y_pred

    #

    def save(self, aPath):
        torch.save(self.state_dict(), aPath)
        # TO LOAD: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()

    #


#
#
#


class THistory:
    def __init__(self):
        self.history = None


#
#
#
#
#
#


class LSTM_PT(TModel.LSTM):
    #
    def __init__(self, fileIN=None, args=[]):
        super(LSTM_PT, self).__init__(fileIN, args)

    #

    def build_model(self):
        model = LSTM(
            features=self.df.shape[1],
            time_steps=self.pars["timeSteps"],
            cells=self.pars["cells"],
            layers=self.pars["layers"],
            output_dim=self.pars["predictionVectorLength"],
            dropout=self.pars["dropout"] * 1e-2,
        )
        return model

    #

    def getRootDirName(self):
        retVal = os.path.abspath(
            os.path.join(
                getModelDirName(self.pars["version"]),
                self.pars["fund"].replace(" ", "_"),
            )
        )
        return retVal

    #


#
#


def getModelDirName(version):
    retVal = None
    if version > 0:
        retVal = os.path.abspath(
            os.path.join(TModel.getRootData(), "model_v_%03d_PT" % (version))
        )
    else:
        retVal = os.path.abspath(os.path.join(TModel.getRootData(), "model"))
    return retVal


#
#############
#
# eof
