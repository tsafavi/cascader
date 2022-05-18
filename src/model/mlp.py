import torch


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.dropouts.append(torch.nn.Dropout(p=dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        for i in range(len(self.lins)):
            torch.nn.init.xavier_uniform_(self.lins[i].weight)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin, dropout in zip(self.lins[:-1], self.dropouts):
            x = lin(x)
            # x = torch.relu(x)
            x = dropout(x)
        x = self.lins[-1](x)
        return x