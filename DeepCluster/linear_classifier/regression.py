
import torch.nn as nn



class logistic_regression(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, conv, num_labels):
        super(logistic_regression, self).__init__()
        self.conv = conv
        if conv==1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv==2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv==3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)
