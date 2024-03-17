from torch import nn

class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_size
    ):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            in_features, hidden_size,
            bidirectional = True,
            batch_first = True
        )
        self.out = nn.Linear(hidden_size * 2, out_features)

    def forward(self, x):
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return self.out(output)