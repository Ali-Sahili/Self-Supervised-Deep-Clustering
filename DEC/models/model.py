import torch
import torch.nn as nn

from models.clustering import Soft_ClusterAssignment


class DEC(nn.Module):
    def __init__(self, cluster_number, hidden_dimension, encoder, alpha = 1.0):
        """
        This module includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = Soft_ClusterAssignment(cluster_number, self.hidden_dimension, alpha)

    def forward(self, batch):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.encoder(batch))
