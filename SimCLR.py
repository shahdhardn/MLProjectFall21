import lightly
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim

from data import calculate_cls_weight
from train import train_nn


def SimCLR(device, train_loader, test_loader, trainData ):
    num_ftrs = 32
    resnet18_new = torchvision.models.resnet18()
    last_conv_channels = list(resnet18_new.children())[-1].in_features
    # note that we need to create exactly the same backbone in order to load the weights
    backbone_new = nn.Sequential(
        *list(resnet18_new.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
    )
    model_all = lightly.models.SimCLR(backbone_new, num_ftrs=num_ftrs)
    ckpt = torch.load('simclr_model_all.pth')
    model_all.load_state_dict(ckpt['resnet18_parameters'])
    model_all.projection_head = nn.Linear(in_features=32, out_features=4, bias=True)
    model_all = model_all.to(device)

    loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(trainData))
    loss_func_test = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model_all.parameters(), lr=0.001)

    epochs = 25
    modelname = 'resnet18_simclr_pretrained_downstream'
    model, _, _, _, train_loss, test_loss = train_nn(model_all,
                                                     train_loader, test_loader,
                                                     loss_func_train, loss_func_test, optimizer, epochs, modelname)

    num_ftrs = 32
    resnet18_new = torchvision.models.resnet18()
    last_conv_channels = list(resnet18_new.children())[-1].in_features
    # note that we need to create exactly the same backbone in order to load the weights
    backbone_new = nn.Sequential(
        *list(resnet18_new.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
    )
    model_all = lightly.models.SimCLR(backbone_new, num_ftrs=num_ftrs)
    model_all.projection_head = nn.Linear(in_features=32, out_features=4, bias=True)

    ckpt = torch.load('resnet18_simclr_pretrained_downstream.pth')
    model_all.load_state_dict(ckpt)

    return model_all, loss_func_test
