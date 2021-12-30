from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
import torch.nn as nn
from utils import str2bool
from train import train_model
from network.model import DCEC
from torchvision import transforms, datasets




def get_args():
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    # Dataset
    parser.add_argument('--data_path', type=str, default='./data/', help='dataset location')
    
    # Model parameters
    parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--bn', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    # Training parameters
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for clustering')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma')
    parser.add_argument('--num_epochs', default=1000, type=int, help='clustering epochs')
    parser.add_argument('--print_freq', default=10, type=int, help='printing frequency')
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--update_interval', default=80, type=int, help='update interval')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    
    args = parser.parse_args()
    
    return args


# Main function
def main(args):

    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data transformations
    data_transforms = transforms.Compose([  transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                         ])

    # Prepare and load Training dataset
    train_set = datasets.MNIST(args.data_path, train=True, download=True, 
                                                           transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, 
                                               shuffle=True, num_workers=args.num_workers)
    trainset_size = len(train_set)
    print("Train set size: ", trainset_size)
    
    # Prepare and load Testing dataset
    #test_set = datasets.MNIST(args.data_path, train=False, download=True, 
     #                                                      transform=data_transforms)

    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, 
     #                                         shuffle=False, num_workers=args.num_workers)
    #testset_size = len(test_set)
    #print("Test set size: ", testset_size)
    print()
    
    # Example batch
    example_batch,_ = next(iter(train_loader))
    
    img_size = example_batch.shape[2]
    n_channels = example_batch.shape[1]
    
    # Define the model
    model = DCEC(args, input_shape=[img_size, img_size, n_channels])
    model = model.to(device)
    
    # Reconstruction loss
    criterion_1 = nn.MSELoss(size_average=True)
    
    # Clustering loss
    criterion_2 = nn.KLDivLoss(size_average=False)

    criterion = [criterion_1, criterion_2]

    # Define the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=args.lr, weight_decay=args.weight_decay)
    # Scheduling
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, 
                                                           gamma=args.sched_gamma)
    # Training 
    train_model(args, model, train_loader, criterion, optimizer, scheduler, device)

        
if __name__ == "__main__":
    args = get_args()
    main(args)
