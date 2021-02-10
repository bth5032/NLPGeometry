# Standard libraries

# External imports
import torch
import torch.nn.functional as F


def train_epoch(model, device, train_loader, optimizer, epoch,
                log_interval=100):
    """Bare bones training loop.
    """

    model.train()

    total_loss = 0
    instances_counted =0
    mean_losses = []
    for instance_idx, (data, target) in enumerate(train_loader):

        instances_counted +=1

        data, target = data.to(device), target.to(device)

        # Make target int a tensor
        target = target.clone().detach().reshape(1)

        # zero gradients
        optimizer.zero_grad()

        # compute output
        output = model(data)

        # compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)

        # add to running total
        total_loss += loss.item()

        # chain rule
        loss.backward()

        # descent step
        optimizer.step()

        if instance_idx % log_interval == 0:
            mean_loss = total_loss/instances_counted
            mean_losses.append(mean_loss)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Running Mean Loss: {:.6f}'.format(
                epoch, instance_idx*len(data) , len(train_loader.dataset),
                100. * instance_idx / len(train_loader), mean_loss))

    return mean_losses

def val_epoch(model, device, val_loader, epoch, log_interval=100):
    """Bare bones evaluation loop.
    """

    model.eval()

    total_loss = 0
    instances_counted = 0
    mean_losses = []
    with torch.no_grad():
        for instance_idx, (data, target) in enumerate(val_loader):

            instances_counted +=1

            data, target = data.to(device), target.to(device)

            # Make target int a tensor
            target = target.clone().detach().reshape(1)

            # compute output
            output = model(data)

            # compute loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, target)

            # add to running total
            total_loss += loss.item()

            if instance_idx % log_interval == 0:
                mean_loss = total_loss/instances_counted
                mean_losses.append(mean_loss)
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\t Running Mean Loss: {:.6f}'.format(
                    epoch, instance_idx*len(data) , len(val_loader.dataset),
                    100. * instance_idx / len(val_loader), mean_loss))

    return mean_losses
