from collections import OrderedDict
import lightly
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.optim as optim

from train import train_nn
from data import calculate_cls_weight

class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, 4)


    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

def MoCo(device, train_loader, test_loader, trainData):
    resnet = lightly.models.ResNetGenerator('resnet-18', 1)

    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.AdaptiveAvgPool2d(1),
    )

    resnet_moco = lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)

    ckpt = torch.load('model_moco.pth')
    new_ckpt = OrderedDict()
    for key, value in ckpt['resnet18_parameters'].items():
        key = key.replace('resnet_moco.', '')
        new_ckpt[key] = value

    resnet_moco.load_state_dict(new_ckpt)

    classifier = Classifier(resnet_moco.backbone)

    loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(trainData))
    loss_func_test = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(classifier.parameters(), lr=0.001)

    classifier = classifier.to(device)
    epochs = 25
    modelname = 'resnet18_moco_pretrained_downstream'
    model, _, _, _, train_loss, test_loss = train_nn(classifier,
                                                     train_loader, test_loader,
                                                     loss_func_train, loss_func_test, optimizer, epochs, modelname)

    resnet = lightly.models.ResNetGenerator('resnet-18', 1)

    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.AdaptiveAvgPool2d(1),
    )

    resnet_moco = lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)
    best_model = Classifier(resnet_moco.backbone)

    ckpt = torch.load('checkpoint/resnet18_moco_pretrained_downstream.pth')
    best_model.load_state_dict(ckpt)

    loss_func_test = nn.CrossEntropyLoss()
    best_model = best_model.to(device)

    return best_model, loss_func_test
