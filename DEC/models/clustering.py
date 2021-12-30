import torch
import torch.nn as nn


class Soft_ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha = 1.0, cluster_centers = None):
        """
        Module to handle the soft assignment, where the Student's t-distribution is used measure 
        similarity between feature vector and each cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(Soft_ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, 
                                                  self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, batch):
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of 
        assignments for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
