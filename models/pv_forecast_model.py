import torch
from torch import nn


class PVForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, nr_layers, output_size):
        super(PVForecastModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=nr_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_activation_fn = nn.Tanh()

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # self.loss_fn = nn.MSELoss()

    def forward(self, x):
        rnn_out, _ = self.lstm(x)
        last_hidden = rnn_out[:, -1, :]
        out = self.fc(last_hidden)
        out = self.output_activation_fn(out)
        return out

    # def train_on(self, X, Y):
    #     self.train()
    #     self.optimizer.zero_grad()
    #     outputs = self(X)
    #     # MSE loss
    #     loss = self.loss_fn(outputs, Y)
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     return loss
    #
    # def eval(self, X, Y):
    #     self.eval()
    #     with torch.no_grad():
    #         outputs = self(X)
    #         loss = self.loss_fn(outputs, Y)
    #     return loss
