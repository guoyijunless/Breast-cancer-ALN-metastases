import math
from numpy import size
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
from MSModel import MSModel

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.classifier = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(), nn.Linear(hidden_size, output_size))

    def forward(self, input):
        input = torch.permute(input, (1,0,2))
        output, hidden = self.rnn_layer(input)
        # output = torch.transpose(output, 0, 1)
        # output = torch.flatten(output, 1, -1)
        pred = self.classifier(output[-1])

        return pred

    def initHidden(self, batchsz):
        return torch.zeros(1, batchsz, self.hidden_size)

class CRNN(nn.Module):
    def __init__(self, in_planes, input_size, hidden_size, num_layers, output_size):
        super(CRNN, self).__init__()
        self.extractor = MSModel(in_planes)
        self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = torch.permute(x, [1, 0, 2, 3, 4])
        s = []
        for i in range(len(x)):
            s.append(self.extractor(x[i]))
        s = torch.stack(s, dim=0)
        output, hidden = self.rnn_layer(s)
        pred = self.classifier(output[-1])

        return pred


class CRNNProto(nn.Module):
    def __init__(self, cnn, rnn, output_size):
        super().__init__()
        self.extractor = cnn
        self.rnn_layer = rnn
        self.classifier = nn.Linear(rnn.hidden_size, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = torch.permute(x, [1, 0, 2, 3, 4])
        s = []
        for i in range(len(x)):
            s.append(self.extractor(x[i]))
        s = torch.stack(s, dim=0)
        output, hidden = self.rnn_layer(s)
        pred = self.classifier(output[-1])

        return pred

if __name__ == '__main__':
    
    model = CRNN(1, 496, 496, 4, 2)
    x  = torch.rand([24, 6, 1, 224, 224])
    print(model(x).shape)
