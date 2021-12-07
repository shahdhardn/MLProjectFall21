import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
from skimage import io

def pretrainSimCLR():
    path_to_chex = '/home/shahad.hardan/Documents/covid19-radiography-database/CheXpert-v1.0-small/train/'
    for i in os.listdir(path_to_chex):
        c = 0
        for j in os.listdir(path_to_chex + i):
            for k in os.listdir(path_to_chex + i + '/' + j):
                c += 1
                img = io.imread(path_to_chex + i + '/' + j + '/' + k)
                io.imsave("/home/shahad.hardan/Downloads/ML701Prj/Chexpert_simclr/" + i + '_' + str(c) + '.png', img)

    num_workers = 8
    batch_size = 256
    seed = 1
    max_epochs = 50
    input_size = 224
    num_ftrs = 32

    pl.seed_everything(seed)

    path_to_data = '/home/shahad.hardan/Downloads/ML701Prj/Chexpert_simclr'

    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        vf_prob=0.5,
        hf_prob=0.5,
        rr_prob=0.5,
        gaussian_blur=0,
        cj_prob=0.0,
        random_gray_scale=0.0)

    dataset_train_simclr = lightly.data.LightlyDataset(
        input_dir=path_to_data)

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers)

    resnet = torchvision.models.resnet18()
    last_conv_channels = list(resnet.children())[-1].in_features
    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1))

    # create the SimCLR model using the newly created backbone
    model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)

    criterion = lightly.loss.NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    encoder = lightly.embedding.SelfSupervisedEmbedding(
        model,
        criterion,
        optimizer,
        dataloader_train_simclr)

    gpus = 1 if torch.cuda.is_available() else 0

    encoder.checkpoint_callback.save_last = True
    encoder.checkpoint_callback.save_top_k = 1

    encoder.train_embedding(gpus=gpus,
                            progress_bar_refresh_rate=100,
                            max_epochs=max_epochs)

    pretrained_resnet_backbone = model.backbone

    state_dict = {'resnet18_parameters': pretrained_resnet_backbone.state_dict()}

    torch.save(state_dict, 'simclr_model.pth')

    resnet18_new = torchvision.models.resnet18()
    last_conv_channels = list(resnet.children())[-1].in_features
    # note that we need to create exactly the same backbone in order to load the weights
    backbone_new = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
    )

    ckpt = torch.load('simclr_model.pth')
    backbone_new.load_state_dict(ckpt['resnet18_parameters'])

    len(os.listdir("/home/shahad.hardan/Downloads/ML701Prj/Chexpert_simclr"))

    pretrained_resnet = model

    state_dict = {'resnet18_parameters': pretrained_resnet.state_dict()}

    torch.save(state_dict, 'simclr_model_all.pth')

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



