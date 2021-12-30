import torch
import torch.nn as nn



# Clustering layer definition (see DCEC article for equations)
class Clustering_Layer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(Clustering_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return ('in_features={}, out_features={}, alpha={}'.format(self.in_features, 
                                                                   self.out_features, self.alpha))

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
