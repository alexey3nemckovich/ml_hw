import torch


def unet_loss(preds, targets):
    ce_loss = torch.nn.CrossEntropyLoss()(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc
