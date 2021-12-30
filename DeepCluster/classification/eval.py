import torch
import numpy as np
from sklearn import metrics




def evaluate(loader, model, eval_random_crops):
    model.eval()
    gts = []
    scr = []
    for crop in range(9 * eval_random_crops + 1):
        for i, (input, target) in enumerate(loader):
            # move input to gpu and optionally reshape it
            if len(input.size()) == 5:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w)
            input = input.cuda(non_blocking=True)

            # forward pass without grad computation
            with torch.no_grad():
                output = model(input)
            if crop < 1 :
                    scr.append(torch.sum(output, 0, keepdim=True).cpu().numpy())
                    gts.append(target)
            else:
                    scr[i] += output.cpu().numpy()
    gts = np.concatenate(gts, axis=0).T
    scr = np.concatenate(scr, axis=0).T
    aps = []
    for i in range(20):
        # Subtract eps from score to make AP work for tied scores
        ap = metrics.average_precision_score(gts[i][gts[i]<=1], scr[i][gts[i]<=1]-1e-5*gts[i][gts[i]<=1])
        aps.append( ap )
    print(np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]))

