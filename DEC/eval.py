import tqdm
import torch



def predict(args, data_iterator, model, use_cuda = True, return_actual = False):
    """
    Predict clusters for a dataset given a DEC model instance and various configuration 
    parameters.
    """

    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError("Dataset has no actual value to unpack, but return_actual is set.")
        if use_cuda:
            batch = batch.cuda(non_blocking=True)
        
        # move to the CPU to prevent out of memory on the GPU
        features.append(model(batch).detach().cpu())  
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features).max(1)[1]
