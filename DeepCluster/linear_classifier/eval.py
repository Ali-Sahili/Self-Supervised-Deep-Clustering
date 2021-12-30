import time
import torch
import torch.nn as nn
from utils import accuracy, AverageMeter, forward



def validate(args, val_loader, model, logistic_reg, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    
    for i, (input_tensor, target) in enumerate(val_loader):
        if args.tencrops:
            bs, ncrops, c, h, w = input_tensor.size()
            input_tensor = input_tensor.view(-1, c, h, w)
        # Input
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Output
        output = logistic_reg(forward(input_var, model, logistic_reg.conv))

        if args.tencrops:
            output_central = output.view(bs, ncrops, -1)[: , ncrops / 2 - 1, :]
            output = softmax(output)
            output = torch.squeeze(output.view(bs, ncrops, -1).mean(1))
        else:
            output_central = output

        # Compute accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input_tensor.size(0))
        top5.update(prec5[0], input_tensor.size(0))
        loss = criterion(output_central, target_var)
        losses.update(loss.data[0], input_tensor.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 100 == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
