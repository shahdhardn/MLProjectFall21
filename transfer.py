import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch

from data import calculate_cls_weight
from train import train_nn

def transferImageNetDENSNET(device, train_loader, test_loader, trainData, epochs):
    d_net = models.densenet121(pretrained=True)
    num_ftrs_d = d_net.classifier.in_features
    num_classes = 4
    d_net.classifier = nn.Linear(num_ftrs_d, num_classes)
    d_net = d_net.to(device)

    loss_func_test = nn.CrossEntropyLoss()

    loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(trainData))

    optimizer = optim.AdamW(d_net.parameters(), lr=0.001)

    modelname = 'densenet121_preimagenet_ftcxr'
    epochs = epochs
    model, _, _, _, train_loss, test_loss = train_nn(d_net,
                                                     train_loader, test_loader,
                                                     loss_func_train, loss_func_test, optimizer, epochs, modelname)

    return num_classes, loss_func_test, train_loss, test_loss

def transferImageNetRESNET(device, train_loader, test_loader, trainData, epochs):
    # Load the best pretrained model
    best_model = models.resnet18()
    best_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    input_ftrs = best_model.fc.in_features
    num_classes = 6
    best_model.fc = nn.Linear(input_ftrs, num_classes)

    ckpt = torch.load('checkpoint/resnet18_prechexpert.pth')
    best_model.load_state_dict(ckpt)

    input_ftrs = best_model.fc.in_features
    # number of output classes
    num_classes = 4
    # replace the fully connected layer to make it comaptible with our datset
    best_model.fc = nn.Linear(input_ftrs, num_classes)

    # transfer the model to gpu
    best_model = best_model.to(device)

    # instantiate crossentropy loss object
    loss_func_test = nn.CrossEntropyLoss()

    loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(trainData))

    optimizer = optim.AdamW(best_model.parameters(), lr=0.001)

    modelname = 'resnet18_prechexpert_ftcxr'
    epochs = epochs  ## args.epochs
    model, _, _, _, train_loss, test_loss = train_nn(best_model,
                                                     train_loader, test_loader,
                                                     loss_func_train, loss_func_test, optimizer, epochs, modelname)


    return num_classes, loss_func_test, train_loss, test_loss


def transferCheXpertDENSNET(device, train_loader, test_loader, trainData, epochs):
    def transferCheXpertRESNET(device, train_loader, test_loader, trainData, epochs):