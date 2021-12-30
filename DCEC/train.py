import copy
import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from utils import metrics, tensor2img


# K-means clusters initialisation
def kmeans(model, dataloader, device):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(device)
        _, _, outputs = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 50000: break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(device))
    # torch.cuda.empty_cache()


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, device):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, outputs, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, label_array, preds


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist


# Training function
def train_model(args, model, dataloader, criterions, optimizer, scheduler, device):

    dataset_size = len(dataloader.dataset)

    # Note the time
    start_time = time.time()

    # Initialise clusters
    kmeans(model, copy.deepcopy(dataloader), device)

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution
    output_distribution, labels, preds_prev = calculate_predictions(model, 
                                                                    copy.deepcopy(dataloader), 
                                                                    device)
    target_distribution = target(output_distribution)
    nmi = metrics.nmi(labels, preds_prev)
    ari = metrics.ari(labels, preds_prev)
    acc = metrics.acc(labels, preds_prev)
    print(f'Metrics: NMI: {nmi} | ARI: {ari} | Accuracy: {acc} |')

    # Go through all epochs
    print("Start training...")
    for epoch in range(args.num_epochs):

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0

        # Iterate over data.
        for batch_num, data in enumerate(dataloader, 1):
            # Get the inputs and labels
            inputs, _ = data

            inputs = inputs.to(device)

            # Uptade target distribution, chack and print performance
            if (batch_num-1) % args.update_interval==0 and not (batch_num==1 and epoch == 0):
                output_distribution, labels, preds = calculate_predictions(model, dataloader, 
                                                                                       device)
                target_distribution = target(output_distribution)
                nmi = metrics.nmi(labels, preds)
                ari = metrics.ari(labels, preds)
                acc = metrics.acc(labels, preds)

                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < args.tol:
                    print('Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    print('Reached tolerance threshold. Stopping training.')
                    break

            tar_dist = target_distribution[((batch_num - 1) * args.batch_size):(batch_num*args.batch_size), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # print(tar_dist)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _ = model(inputs)
                loss_rec = criterions[0](outputs, inputs)
                loss_tmp = criterions[1](torch.log(clusters), tar_dist)
                loss_clust = args.gamma * loss_tmp / args.batch_size
                loss = loss_rec + loss_clust
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_clust.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * args.batch_size + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * args.batch_size + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * args.batch_size + inputs.size(0))

            if batch_num % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss clustering {7:.4f} ({8:.4f})\t'.format(epoch+1, 
                                           batch_num, len(dataloader), 
                                           loss_batch, loss_accum, loss_batch_rec, 
                                           loss_accum_rec, loss_batch_clust, loss_accum_clust))

            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader) and (epoch+1) % 5:
                inp = tensor2img(inputs)
                out = tensor2img(outputs)
                
                #img = np.concatenate((inp, out), axis=1)
                

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        print('Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(
                                                epoch_loss, epoch_loss_rec, epoch_loss_clust))

        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save final model
    torch.save(best_model_wts, 'checkpoint.pth')

