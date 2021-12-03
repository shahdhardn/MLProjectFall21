import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from data import calculate_cls_weight
from train import train_nn

def baselineDENSNET(device, train_loader, test_loader, trainData, epochs):
    d_net = models.densenet121(pretrained=False)
    # d_net.classifier.in_features
    num_ftrs_d = d_net.classifier.in_features
    num_classes = 4
    d_net.classifier = nn.Linear(num_ftrs_d, num_classes)
    d_net = d_net.to(device)

    loss_func_test = nn.CrossEntropyLoss()

    loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(trainData))

    optimizer = optim.AdamW(d_net.parameters(), lr=0.001)

    epochs = epochs
    modelname = 'densenet121_baseline'
    model, _, _, _, train_loss, test_loss = train_nn(d_net,
                                                     train_loader, test_loader, loss_func_train,
                                                     loss_func_test, optimizer, epochs, modelname, device)

    return num_classes, loss_func_test, train_loss, test_loss

def baselineRESNET(device, train_loader, test_loader, trainData, epochs):
    # instantiate resenet18 model object
    resnet18_model = models.resnet18(pretrained=False)
    # Fully connected layer input features:
    input_ftrs = resnet18_model.fc.in_features
    # number of output classes
    num_classes = 4
    # replace the fully connected layer to make it comaptible with our datset
    resnet18_model.fc = nn.Linear(input_ftrs, num_classes)
    # transfer the model to gpu
    resnet18_model = resnet18_model.to(device)
    # #instantiate crossentropy loss object

    loss_func_test = nn.CrossEntropyLoss()

    loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(trainData))

    optimizer = optim.AdamW(resnet18_model.parameters(), lr=0.001)

    epochs = epochs
    modelname = 'resnet18_baseline'
    model, _, _, _, train_loss, test_loss = train_nn(resnet18_model,
                                                     train_loader, test_loader,
                                                     loss_func_train, loss_func_test, optimizer, epochs, modelname)

    return num_classes, loss_func_test, train_loss, test_loss