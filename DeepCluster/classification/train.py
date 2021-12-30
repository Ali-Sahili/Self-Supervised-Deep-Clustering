import time
import torch





def train(args, data_loader, model, optimizer, criterion, losses, it=0, verbose=True):
    # to log
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    current_iteration = it

    # use dropout for the MLP
    model.train()
    # in the batch norms always use global statistics
    model.features.eval()

    for (input, target) in loader:
        # measure data loading time
        data_time.update(time.time() - end)
        
        # adjust learning rate
        if current_iteration != 0 and current_iteration % args.stepsize == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                print('iter {0} lr is {1}'.format(current_iteration, param_group['lr']))

        # move input to gpu
        input = input.cuda(non_blocking=True)

        # forward pass with or without grad computation
        output = model(input)

        target = target.float().cuda()
        mask = (target == 255)
        loss = torch.sum(criterion(output, target).masked_fill_(mask, 0)) / target.size(0)

        # backward 
        optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # and weights update
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if verbose is True and current_iteration % 25 == 0:
            print('Iteration[{0}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   current_iteration, batch_time=batch_time,
                   data_time=data_time, loss=losses))
        current_iteration = current_iteration + 1
        if args.nit is not None and current_iteration == args.nit:
            break
    return current_iteration
