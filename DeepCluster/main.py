import warnings
warnings.filterwarnings('ignore')

import os
import time
import faiss
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.metrics.cluster import normalized_mutual_info_score

import models
import clustering
from helpers import *
from utils import Logger
from train import train_one_epoch


# Settings parameters
def get_parser():
    parser = argparse.ArgumentParser('DeepCluster Algorithm', add_help=False)

    parser.add_argument('--data', type=str, default='tiny-imagenet-200/train', 
                                          help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, choices=['alexnet', 'vgg16'], 
                         default='alexnet', help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="nb of epochs between two consecutive reassignments of clusters.")
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', type=int, default=200, help='nb of total epochs to run.')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint ')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='nb of iters between two checkpoints')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    
    return parser

# Main function
def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = model.features
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                                lr=args.lr, momentum=args.momentum, weight_decay=10**args.wd)

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint '{}' (epoch {})".format(args.resume,checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])
    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=train_transforms)
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(args, dataloader, model, len(dataset))

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        train_set = clustering.cluster_assign(deepcluster.images_lists, dataset.imgs)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign*len(train_set)), deepcluster.images_lists)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   sampler=sampler, pin_memory=True)

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train_one_epoch(args, train_loader, model, criterion, optimizer, epoch)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
