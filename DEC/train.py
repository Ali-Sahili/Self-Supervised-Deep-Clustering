import tqdm
from utils import target_distribution



""" train function for ope epoch """
def train_one_epoch(args, data_iterator, model, optimizer, loss_function, features,
                     epoch, delta_label, accuracy, use_cuda=True):

    model.train()
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, _ = batch  # if we have a prediction label, strip it away
        if use_cuda:
            batch = batch.cuda(non_blocking=True)
        output = model(batch)
        target = target_distribution(output).detach()
        loss = loss_function(output.log(), target) / output.shape[0]
        data_iterator.set_postfix( epo=epoch, acc="%.4f" % (accuracy or 0.0),
                                              lss="%.8f" % float(loss.item()), 
                                              dlb="%.4f" % (delta_label or 0.0))
            
        # Back-prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure=None)
            
        features.append(model.encoder(batch).detach().cpu())
        if args.update_freq is not None and index % args.update_freq == 0:
            loss_value = float(loss.item())
            data_iterator.set_postfix(epo=epoch, acc="%.4f" % (accuracy or 0.0),
                                                 lss="%.8f" % loss_value,
                                                 dlb="%.4f" % (delta_label or 0.0))

    return  features
