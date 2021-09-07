import numpy as np
import torch


def islandinfo(y, trigger_val, stopind_inclusive=True):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return list(zip(idx[:-1:2], idx[1::2]-int(stopind_inclusive))), lens


def get_accuracy(model, data_iter, batch_size):
    with torch.no_grad():
        model.eval()
        batch_iter = 0
        num_correct = 0
        for x, y in data_iter:
            x = x.transpose(0,1).cuda()
            # x = x.cuda()

            y_hat = model(x)

            y_hat = torch.argmax(y_hat, dim=1)

            num_correct += torch.sum(torch.eq(y_hat, y.cuda())).item() #y.cuda()

            batch_iter += 1

        accuracy = num_correct / (batch_iter * batch_size)

    return accuracy
