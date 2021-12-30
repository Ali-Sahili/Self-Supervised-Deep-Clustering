import time
import torch
from utils import AverageMeter



def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch):
    """Training of the CNN.
        Args:
            train_loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            criterion (torch.nn): loss
            optimizer (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(model.top_layer.parameters(),
                                   lr=args.lr, weight_decay=10**args.wd)

    end = time.time()
    for i, (input_tensor, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(train_loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(args.exp, 'checkpoints',
                                          'checkpoint_' + str(n/args.checkpoints) + '.pth.tar')
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.data[0], input_tensor.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg
