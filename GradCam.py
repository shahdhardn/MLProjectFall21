from MoCo import Classifier
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models
import torch.nn as nn
import torch
from skimage import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage import color
import matplotlib
import lightly


font={'weight':'bold', 'size':11}
matplotlib.rc('font', **font)

def GradCam(device, modelname):
    if(modelname=='TransferResnet'):
        # Transfer learning| ReseNet18
        model_tl = models.resnet18()
        input_ftrs = model_tl.fc.in_features
        num_classes = 4
        # replace the fully connected layer to make it comaptible with our datset
        model_tl.fc = nn.Linear(input_ftrs, num_classes)

        ckp = torch.load('checkpoint/resnet18_prechexpert_ftcxr.pth')

        model_tl.load_state_dict(ckp)
        model_tl = model_tl.to(device)

        img_org = io.imread('COVID_resized224/Indeterminate/6c7422f3cd88.png').astype('float32')
        img_org = img_org / np.max(img_org)
        img_pil = Image.open('COVID_resized224/Indeterminate/6c7422f3cd88.png')
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img_pil)
        img = img.cuda()
        img.dtype
        img_org.dtype
        img_org.max()
        cam = GradCAM(model=model_tl, target_layers=[model_tl.layer4[-1]])
        grayscale_cam = cam(input_tensor=img[None], target_category=1)
        grayscale_cam = grayscale_cam[0, :]
        new_image = color.gray2rgb(img_org)

        visualization = show_cam_on_image(new_image, grayscale_cam)
        plt.imshow(visualization, cmap="gray")

    elif(modelname=='TransferDensenet'):
        # Transfer learning| DenseNet121
        d_net = models.densenet121()
        num_ftrs_d = d_net.classifier.in_features
        num_classes = 4
        d_net.classifier = nn.Linear(num_ftrs_d, num_classes)
        ckp = torch.load('checkpoint/densenet121_prechexpert_ftcxr.pth')
        d_net.load_state_dict(ckp)
        d_net = d_net.to(device)
        img_org = io.imread('COVID_resized224/Indeterminate/6c7422f3cd88.png').astype('float32')
        img_org = img_org / np.max(img_org)
        img_pil = Image.open('COVID_resized224/Indeterminate/6c7422f3cd88.png')
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img_pil)
        img = img.cuda()
        img.dtype
        img_org.dtype
        img_org.max()
        cam = GradCAM(model=d_net, target_layers=d_net.features.denseblock4.denselayer16.conv2)
        grayscale_cam = cam(input_tensor=img[None], target_category=1)
        grayscale_cam = grayscale_cam[0, :]
        new_image = color.gray2rgb(img_org)

        visualization = show_cam_on_image(new_image, grayscale_cam)
        plt.imshow(visualization, cmap="gray")

    elif(modelname=='SimCLR'):
        num_ftrs = 32
        resnet18_new = models.resnet18()
        last_conv_channels = list(resnet18_new.children())[-1].in_features
        # note that we need to create exactly the same backbone in order to load the weights
        backbone_new = nn.Sequential(
            *list(resnet18_new.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
        )
        model_simclr = lightly.models.SimCLR(backbone_new, num_ftrs=num_ftrs)
        model_simclr.projection_head = nn.Linear(in_features=32, out_features=4, bias=True)

        ckpt = torch.load('checkpoint/resnet18_simclr_pretrained_downstream_u.pth')
        model_simclr.load_state_dict(ckpt)

        model_simclr = model_simclr.to(device)
        img_org = io.imread('COVID_RGB/Atypical/2cb9a2a71fac.png').astype('float32')
        img_org = img_org / np.max(img_org)
        img_pil = Image.open('COVID_RGB/Atypical/2cb9a2a71fac.png')
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img_pil)
        img = img.cuda()
        img.dtype
        img_org.dtype
        img_org.max()
        cam = GradCAM(model=model_simclr, target_layers=[model_simclr.backbone[7][-1]])
        grayscale_cam = cam(input_tensor=img[None], target_category=0)
        grayscale_cam = grayscale_cam[0, :]
        new_image = color.gray2rgb(img_org)

        visualization = show_cam_on_image(new_image, grayscale_cam)
        plt.imshow(visualization, cmap="gray")

    elif(modelname=='MoCo'):
        resnet = lightly.models.ResNetGenerator('resnet-18', 1)

        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        resnet_moco = lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)
        best_model = Classifier(resnet_moco.backbone)

        ckpt = torch.load('checkpoint/resnet18_moco_pretrained_downstream.pth')
        best_model.load_state_dict(ckpt)

        best_model = best_model.to(device)

        img_org = io.imread('COVID_RGB/Atypical/5ea54f3cfdbb.png').astype('float32')
        img_org = img_org / np.max(img_org)
        img_pil = Image.open('COVID_RGB/Atypical/5ea54f3cfdbb.png')
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img_pil)
        img = img.cuda()
        img.dtype
        img_org.dtype
        img_org.max()
        cam = GradCAM(model=best_model, target_layers=[best_model.backbone[5][-1]])
        grayscale_cam = cam(input_tensor=img[None], target_category=0)
        grayscale_cam = grayscale_cam[0, :]
        new_image = color.gray2rgb(img_org)

        visualization = show_cam_on_image(new_image, grayscale_cam)
        plt.imshow(visualization, cmap="gray")














