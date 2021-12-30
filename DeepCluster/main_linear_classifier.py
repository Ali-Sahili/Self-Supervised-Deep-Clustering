import os
import time
import argparse
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import Logger
from helpers import load_model
from linear_classifier.eval import validate
from linear_classifier.train import train_one_epoch
from linear_classifier.regression import logistic_regression


# Prameters Settings
parser = argparse.ArgumentParser(description="Train linear classifier on top of frozen convolutional layers of an AlexNet.")
# Dataset
parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--conv', type=int, choices=[1, 2, 3, 4, 5],
                    help='on top of which convolutional layer train logistic regression')
parser.add_argument('--exp', type=str, default='', help='exp folder to log results.')
# Dataloading
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
# Training
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--epochs', type=int, default=90, help='number of total epochs to run.')
parser.add_argument('--weight_decay', default=-4, type=float, help='weight decay pow')
# 
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--verbose', action='store_true', help='print results')
parser.add_argument('--tencrops', action='store_true',
                    help='validation accuracy averaged over 10 crops')
                    

# Main function to train and evaluate linear classifier 
def main(args):
    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load model
    model = load_model(args.model)
    model.cuda()
    cudnn.benchmark = True

    # freeze the features layers
    for param in model.features.parameters():
        param.requires_grad = False

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # Transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.tencrops:
        transformations_val = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
        ]
    else:
        transformations_val = [transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize]

    transformations_train = [transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize]
    # Prepare datasets
    train_dataset = datasets.ImageFolder(traindir, 
                                         transform=transforms.Compose(transformations_train))

    val_dataset = datasets.ImageFolder(valdir,
                                         transform=transforms.Compose(transformations_val))
    # Dataloading
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True)
                                               
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size/2),
                                             shuffle=False, num_workers=args.num_workers)

    # logistic regression
    logistic_reg = logistic_regression(args.conv, len(train_dataset.classes)).cuda()
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, reglog.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=10**args.weight_decay)

    # create logs
    exp_log = os.path.join(args.exp, 'log')
    if not os.path.isdir(exp_log):
        os.makedirs(exp_log)

    loss_log = Logger(os.path.join(exp_log, 'loss_log'))
    prec1_log = Logger(os.path.join(exp_log, 'prec1'))
    prec5_log = Logger(os.path.join(exp_log, 'prec5'))

    # Training / Validation phase
    print("Start training...")
    best_prec1 = 0
    for epoch in range(args.epochs):
        end = time.time()

        # train for one epoch
        train_one_epoch(args, train_loader, model, logistic_reg, optimizer, criterion, epoch)

        # evaluate on validation set
        prec1, prec5, loss = validate(args, val_loader, model, logistic_reg, criterion)

        loss_log.log(loss)
        prec1_log.log(prec1)
        prec5_log.log(prec5)

        # remember best precision and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            filename = 'model_best.pth.tar'
        else:
            filename = 'checkpoint.pth.tar'
        torch.save({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'state_dict': model.state_dict(),
            'prec5': prec5,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(args.exp, filename))




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
