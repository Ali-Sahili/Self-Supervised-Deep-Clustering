import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from helpers import load_model 
from utils import AverageMeter, accuracy

from classification.train import train
from classification.eval import evaluate
from classification.voc_dataset import VOC2007_dataset

# Settings parameters
parser = argparse.ArgumentParser()
parser.add_argument('--voc_dir', type=str, default='', help='pascal voc 2007 dataset path.')
parser.add_argument('--split', type=str, default='train', choices=['train', 'trainval'], 
                                                          help='training split')
parser.add_argument('--nit', type=int, default=80000, help='Number of training iterations')
parser.add_argument('--fc6_8', type=int, default=1, help='train only the final classifier')
parser.add_argument('--train_batchnorm', type=int, default=0, 
                                         help='train batch-norm layer parameters')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--eval_random_crops', type=int, default=1, 
                     help='If true, eval on 10 random crops, otherwise eval on 10 fixed crops')
parser.add_argument('--stepsize', type=int, default=5000, help='Decay step')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--min_scale', type=float, required=False, default=0.1, help='scale')
parser.add_argument('--max_scale', type=float, required=False, default=0.5, help='scale')
parser.add_argument('--seed', type=int, default=31, help='random seed')


# Main function
def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # create model and move it to gpu
    model = load_model(args.model)
    model.top_layer = nn.Linear(model.top_layer.weight.size(1), 20)
    model.cuda()
    cudnn.benchmark = True

    # what partition of the data to use
    if args.split == 'train':
        args.test = 'val'
    elif args.split == 'trainval':
        args.test = 'test'
        
    # Transformations for training
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.RandomResizedCrop(224, scale=(args.min_scale,args.max_scale), ratio=(1,1)),
          transforms.ToTensor(),
          normalize,
         ])
         
    # Training dataset and loading
    train_set = VOC2007_dataset(args.voc_dir, split=args.split, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, 
                                              shuffle=False, num_workers=4, pin_memory=True)
    print('PASCAL VOC 2007 ' + args.split + ' dataset loaded')

    # Transformations for evaluation
    if args.eval_random_crops:
        transform_eval = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.RandomResizedCrop(224, scale=(args.min_scale,args.max_scale), ratio=(1,1)), 
          transforms.ToTensor(),
          normalize,
        ])
    else:
        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
        ])

    # validation set and loading
    val_dataset = VOC2007_dataset(args.voc_dir, split=args.split, transform=transform_eval)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               num_workers=4,  pin_memory=True)
    # test dataset and dataloading
    test_set = VOC2007_dataset(args.voc_dir, split=args.test, transform=transform_eval)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                               num_workers=4, pin_memory=True)

    # re initialize classifier
    for y, m in enumerate(model.classifier.modules()):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.1)
    model.top_layer.bias.data.fill_(0.1)

    if args.fc6_8:
       # freeze some layers 
        for param in model.features.parameters():
            param.requires_grad = False
        # unfreeze batchnorm scaling
        if args.train_batchnorm:
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = True

    # set optimizer
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                                lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # Define loss function
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Training
    print('Start training')
    it = 0
    losses = AverageMeter()
    while it < args.nit:
        it = train(args, train_loader, model, optimizer, criterion, losses, it=it)

    # Evaluation on validation set
    evaluate(val_loader, model, args.eval_random_crops)
    
    # Evaluation on test set
    evaluate(test_loader, model, args.eval_random_crops)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

