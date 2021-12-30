import warnings
warnings.filterwarnings("ignore")

import uuid
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader, default_collate

import sdae.model as ae
from sdae.sdae import StackedDenoisingAutoEncoder

from eval import predict
from models.model import DEC
from mnist import CachedMNIST
from train import train_one_epoch
from utils import cluster_accuracy, str2bool


# Setting Parameters
def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    
    # Dataset parameters   
    parser.add_argument('--data_location', default='./data', type=str, 
                        help='dataset location - dataset will be downloaded to this folder')   
    # Dataloading parameters
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    # Training parameters
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--evaluate_batch_size', default=1024, type=int, 
                         help='batch size for evaluation stage.')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training.')
    parser.add_argument('--update_freq', default=10, type=int, 
                        help='frequency of batches with which to update counter.')
    parser.add_argument('--cuda', default=True, type=str2bool, help='to use CUDA')
    parser.add_argument('--silent', default=False, type=str2bool, 
                         help='set to True to prevent printing out summary statistics')
    parser.add_argument('--testing_mode', default=False, type=str2bool, 
                         help='whether to run in testing mode')
    parser.add_argument('--stopping_delta', default=0.000001, type=float, 
                        help='label delta as a proportion to use for stopping')
    # Training SAE
    parser.add_argument('--pretrained_epochs', default=100, type=int, help='number of epochs.')
    parser.add_argument('--finetune_epochs', default=100, type=int, help='number of epochs.')
    return parser
   


def main(args):
    """
    Main function to train and evaluate the DEC model on a given dataset.
    """
    # Choose device: CPU/GPU
    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    
    # ==============================================================================
    #                              Datasets & Dataloading
    # ==============================================================================
    # Prepare datasets
    train_set = CachedMNIST(args.data_location, train=True, cuda=use_cuda, 
                                                testing_mode=args.testing_mode)
    val_set = CachedMNIST(args.data_location, train=False, cuda=use_cuda, 
                                              testing_mode=args.testing_mode)
    
    # Dataloading for train and validation datasets
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  collate_fn=default_collate, shuffle=True)
                                  
    test_loader = DataLoader(train_set, batch_size=args.evaluate_batch_size, 
                            collate_fn=default_collate, shuffle=False)
    
    # ==============================================================================
    #                Model initialization - Stacked Auto-Encoder
    # ==============================================================================    
    # Define Auto-encoder to initialize the model encoder
    autoencoder = StackedDenoisingAutoEncoder([28*28,500,500,2000,10], final_activation=None)
    if use_cuda:
        autoencoder.cuda()
        
    print("Pretraining stage.")
    ae.pretrain(train_set, autoencoder, cuda=use_cuda, validation=val_set,
                epochs=args.pretrained_epochs, batch_size=args.batch_size,
                optimizer=lambda model: SGD(model.parameters(), lr=args.lr*10, momentum=0.9),
                scheduler=lambda x: StepLR(x, 100, gamma=0.1), corruption=0.2)
    
    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=args.lr*10, momentum=0.9)
    ae.train(train_set, autoencoder, cuda=use_cuda, validation=val_set, 
             epochs=args.finetune_epochs, batch_size=args.batch_size,
             optimizer=ae_optimizer,
             scheduler=StepLR(ae_optimizer, 100, gamma=0.1), corruption=0.2,
             update_callback=None)
    
    # Define the model
    print("DEC stage.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    if use_cuda:
        model.cuda()
    
    # Define optimizer
    dec_optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Define the KL-divergence loss
    loss_function = nn.KLDivLoss(size_average=False)
    
    # ==============================================================================
    #                  Parameters Initialization - Cluster centroids
    # ==============================================================================
    print("Initializing parameters...")
    static_dataloader = DataLoader(train_set, batch_size=args.batch_size, 
                                   collate_fn=default_collate, pin_memory=False,
                                   shuffle=False)
    
    
    data_iterator = tqdm(static_dataloader, leave=True, unit="batch",
                         postfix={  "epo": -1,
                                    "acc": "%.4f" % 0.0,
                                    "lss": "%.8f" % 0.0,
                                    "dlb": "%.4f" % -1,
                                 },
                         disable=args.silent)
    
    # K-means
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    
    model.train()
    features = []
    actual = []
    
    # Initial cluster centroids
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        if use_cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(model.encoder(batch).detach().cpu())
    actual = torch.cat(actual).long()
    
    # Apply k-means
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    
    # Compute accuracy
    _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    
    cluster_centers = torch.tensor(kmeans.cluster_centers_,dtype=torch.float,requires_grad=True)
    if use_cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
        
    with torch.no_grad():
        # initialise the cluster centroids
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    print("Model's Cluster centroids are initialized.")
    
    
    # ==============================================================================
    #                        Training / Evaluation Phases
    # ==============================================================================
    print("Start training...")
    delta_label = None
    for epoch in range(args.epochs):
        features = []
        
        data_iterator = tqdm(train_loader, leave=True, unit="batch",
                             postfix={  "epo": epoch,
                                        "acc": "%.4f" % (accuracy or 0.0),
                                        "lss": "%.8f" % 0.0,
                                        "dlb": "%.4f" % (delta_label or 0.0),
                                     },
                             disable=args.silent
                         )
        # Training
        features = train_one_epoch(args, data_iterator, model, dec_optimizer, 
                                         loss_function, features, epoch, 
                                         delta_label, accuracy, use_cuda)
        
        # Evaluate
        data_iterator = tqdm(test_loader, leave=True, unit="batch", disable=args.silent)
        predicted, actual = predict(args, data_iterator, model, use_cuda, return_actual = True)
            
            
        delta_label = (float((predicted != predicted_previous).float().sum().item())
                                               / predicted_previous.shape[0]
                      )
        
        if args.stopping_delta is not None and delta_label < args.stopping_delta:
            print('Early stopping as label delta "%1.5f" less than "%1.5f".'
                                                        % (delta_label, args.stopping_delta))
            break
        
        predicted_previous = predicted
        
        # Compute accuracy
        _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
        
        data_iterator.set_postfix(epo=epoch, acc="%.4f" % (accuracy or 0.0),
                                             lss="%.8f" % 0.0,
                                             dlb="%.4f" % (delta_label or 0.0),
                                 )
    print()

    # Evaluate
    model.eval()
    predicted, actual = predict(args, test_loader, model, use_cuda, return_actual = True)
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    
    if not args.testing_mode:
        predicted_reassigned = [reassignment[item] for item in predicted]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (confusion.astype("float") / confusion.sum(axis=1)[:,np.newaxis])
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig("confusion_%s.png"%confusion_id)
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
