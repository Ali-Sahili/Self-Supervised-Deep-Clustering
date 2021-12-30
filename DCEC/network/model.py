import torch
import torch.nn as nn

from network.layers import Clustering_Layer


# Convolutional autoencoder with clustering part
class DCEC(nn.Module):
    def __init__(self, args, input_shape=[128,128,3], filters=[32, 64, 128]):
        super(DCEC, self).__init__()
        
        assert input_shape[0] == input_shape[1]
        
        self.bn = args.bn
        self.leaky = args.leaky
        self.filters = filters
        self.input_shape = input_shape
        self.activations = args.activations
        self.num_clusters = args.num_clusters
        
        bias = args.bias
        
        if self.leaky:
            self.relu = nn.LeakyReLU(negative_slope=args.neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
                
        # Backbone: Features Extractor
        self.features_extrator = nn.Sequential(
                                      nn.Conv2d(input_shape[2], filters[0], 5, 2, 2, bias=bias),
                                      nn.BatchNorm2d(filters[0]) if self.bn else nn.Identity(),
                                      self.relu,
                                      nn.Conv2d(filters[0], filters[1], 5, 2, 2, bias=bias),
                                      nn.BatchNorm2d(filters[1]) if self.bn else nn.Identity(),
                                      self.relu,
                                      nn.Conv2d(filters[1], filters[2], 3, 2, 0, bias=bias)
                                  )
                                  
        # Batch layer at the end of the features extractor
        self.bn_layer = nn.BatchNorm2d(filters[2]) if self.bn else nn.Identity()
        
        # Compute size of output features map
        fc_dim, self.nf = self.Compute_dim()
        
        # Embeddings layers
        self.embedding = nn.Linear(fc_dim, self.num_clusters, bias=bias)
        self.deembedding = nn.Linear(self.num_clusters, fc_dim, bias=bias)
        
        # Compute output padding size
        out_pad_1 = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        out_pad_2 = 1 if input_shape[0] // 2 % 2 == 0 else 0
        out_pad_3 = 1 if input_shape[0] % 2 == 0 else 0
        
        # Deconvolution Layers
        self.deconv_layers = nn.Sequential(
                                    nn.ConvTranspose2d(filters[2], filters[1], 3, 2, 0, 
                                                          output_padding=out_pad_1, bias=bias),
                                    nn.BatchNorm2d(filters[1]) if self.bn else nn.Identity(),
                                    self.relu,
                                    nn.ConvTranspose2d(filters[1], filters[0], 5, 2, 2, 
                                                          output_padding=out_pad_2, bias=bias),
                                    nn.BatchNorm2d(filters[0]) if self.bn else nn.Identity(),
                                    self.relu,
                                    nn.ConvTranspose2d(filters[0], input_shape[2], 5, 2, 2, 
                                                           output_padding=out_pad_3, bias=bias)
                                )
        
        # Clustering method
        self.clustering = Clustering_Layer(self.num_clusters, self.num_clusters)

        # Activation map
        if self.activations:
            self.sig = nn.Sigmoid()
            self.tanh = nn.Tanh()

    def Compute_dim(self):
        # Compute dimensions of features map
        nf0 = self.input_shape[0]
        nf1 = ((nf0 + 4 - 5) // 2) + 1
        nf2 = ((nf1 + 4 - 5) // 2) + 1
        nf3 = ((nf2 + 0 - 3) // 2) + 1
        dim = nf3 * nf3 * self.filters[2]
        return dim, nf3

    def forward(self, x):
        # [-1, nf0, nf0, nc]
        x = self.features_extrator(x)
        
        # [-1, nf3, nf3, filter[2]]
        if self.activations:
            x = self.sig(x)
        else:
            x = self.bn_layer(x)
            x = self.relu(x)
            
        # [-1, nf3, nf3, filter[2]]
        x = x.view(x.size(0), -1)
        # [-1, fc_dim]
        x = self.embedding(x)
        extra_out = x
        
        # [-1, num_clusters]
        clustering_out = self.clustering(x)
        
        # [-1, num_clusters]
        x = self.deembedding(x)
        x = self.relu(x)
        
        # [-1, fc_dim]
        x = x.view(x.size(0), self.filters[2], self.nf, self.nf)
        x = self.deconv_layers(x)
        
        # [-1, nf0, nf0, nc]
        if self.activations:
            x = self.tanh(x)
            
        return x, clustering_out, extra_out


if __name__ == "__main__":
    bs = 1
    model = DCEC()
    x = torch.rand(bs, 3, 128, 128)
    out, clustering_out, extra_out = model(x)
    print(out.shape, clustering_out.shape, extra_out.shape)
