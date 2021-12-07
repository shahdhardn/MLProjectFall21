import argparse

from GradCam import GradCam
from MoCo import MoCo
from MoCo_pretrain import pretrainMoCo
from SimCLR import SimCLR
from SimCLR_pretrain import pretrainSimCLR
from data import get_mean_and_std, classHistograms, show_transformed_images, CXRDataSet
from baseline import baselineDENSNET, baselineRESNET
from analysis import eval_best_model, conf_mtrx, evaluate_metrics, ROC_plot_AUC_score, plot_learning_curve
from preprocessing_train_chexpert import preprocess_train
from test import test_model
from transfer import transferImageNetRESNET, transferImageNetDENSNET, transferCheXpertDENSENET, transferCheXpertRESNET

from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch





def main(args):
    if args.technique == 'baseline RESNET' or 'baseline DENSENET' or 'transfer IMAGENET RESNET' or 'transfer IMAGENET DENSENET' or 'SimCLR' or 'MoCo':
        path_to_rgbdata = args.RGBpath
        data = ImageFolder(path_to_rgbdata)
        #  data variable attributes?
        # print(data)
        print('Total number of datapoints:', len(data.imgs))
        # shows the number of datapoints and root location
        print('---------------------------')
        # print(data.imgs[0:10])
        # list of tuples, each of which contains the image path and its label
        # print(data.targets[0:10])
        # only labels [0,1,2,...]
        print("Dataset classes:", data.classes)
        # the name of the classes [Normal , covid, ..]

        if args.histogram == 'true':
            classHistograms(data)

        # Split dataset into train test sets:
        trainData, testData, trainLabel, testLabel = train_test_split(data.imgs, data.targets, test_size=0.1,
                                                                      random_state=0, stratify=data.targets)
        # get mean and std of training set images
        mean, std = get_mean_and_std(trainData)

        # Image transformations and loading images
        train_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])

        test_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])

        train_set = CXRDataSet(trainData, 'RGB',
                               transform=train_data_transforms)
        test_set = CXRDataSet(testData, 'RGB',
                              transform=test_data_transforms)

        print("Size of train dataset: ", len(train_set))
        print("Size of test dataset: ", len(test_set))


        if (args.showtransformed == 'true'):
            show_transformed_images(train_set, mean, std, data)

    elif args.technique == 'transfer CHEXPERT RESNET' or 'transfer CHEXPERT DENSENET':
        path_to_grayscale = args.GREYpath
        data = ImageFolder(path_to_grayscale)
        print('Total number of datapoints:', len(data.imgs))
        print('---------------------------')
        print("Dataset classes:", data.classes)

        trainData, testData, trainLabel, testLabel = train_test_split(data.imgs, data.targets,
                                                                      test_size=0.1,
                                                                      random_state=0, stratify=data.targets)

        train_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5388]), torch.Tensor([0.1938]))
        ])

        test_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5388]), torch.Tensor([0.1938]))
        ])
        train_set = CXRDataSet(trainData, 'L',
                               transform=train_data_transforms)
        test_set = CXRDataSet(testData, 'L',
                              transform=test_data_transforms)
        print("Size of train dataset: ", len(train_set))
        print("Size of test dataset: ", len(test_set))


    batch_size = args.batchsize  ## Argument
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.is_available()

    seed = 1997
    torch.cuda.manual_seed_all(seed)

    if args.technique == 'baseline DENSENET':
        num_classes, loss_func_test, train_loss, test_loss=baselineDENSNET(device, train_loader, test_loader, trainData, args.epochs)
        # test_labels, pred_cls, pred_proba, t_loss, test_acc =eval_best_model (model, model_path,n_classes ,loss_func_test,device)
        test_labels, pred_cls, pred_proba, _, _ = eval_best_model('densenet121', 'checkpoint/densenet121_baseline.pth',
                                                                  num_classes, loss_func_test, test_loader, device)

    elif args.technique == 'baseline RESNET':
        num_classes, loss_func_test, train_loss, test_loss=baselineRESNET(device, train_loader, test_loader, trainData, args.epochs)
        # test_labels, pred_cls, pred_proba, t_loss, test_acc =eval_best_model (model, model_path,n_classes ,loss_func_test,device)
        test_labels, pred_cls, pred_proba, _, _ = eval_best_model('resnet18', 'checkpoint/resnet18_baseline.pth',
                                                                  num_classes, loss_func_test, test_loader, device)

    elif args.technique == 'transfer IMAGENET RESNET':
        num_classes, loss_func_test, train_loss, test_loss = transferImageNetRESNET(device, train_loader, test_loader,
                                                                            trainData, args.epochs)
        test_labels, pred_cls, pred_proba, _, _ = eval_best_model('resnet18',
                                                                  'checkpoint/resnet18_preimagenet_ftcxr.pth',
                                                                  num_classes, loss_func_test, test_loader, device)


    elif args.technique == 'transfer IMAGENET DENSENET':
        num_classes, loss_func_test, train_loss, test_loss = transferImageNetDENSNET(device, train_loader, test_loader,
                                                                            trainData, args.epochs)
        test_labels, pred_cls, pred_proba, _, _ = eval_best_model('densenet121',
                                                                  'checkpoint/densenet121_preimagenet_ftcxr.pth',
                                                                  num_classes, loss_func_test, test_loader, device)


    elif args.technique == 'transfer CHEXPERT RESNET':
        preprocess_train(device, 'RESNET')
        num_classes, loss_func_test, train_loss, test_loss = transferCheXpertRESNET(device, train_loader, test_loader,
                                                                            trainData, args.epochs)
        test_labels, pred_cls, pred_proba, _, _ = eval_best_model('resnet18',
                                                                  'checkpoint/resnet18_prechexpert_ftcxr.pth',
                                                                  num_classes, loss_func_test, test_loader, device, band='RGB')
        GradCam(device, 'TransferResnet')

    elif args.technique == 'transfer CHEXPERT DENSENET':
        preprocess_train(device, 'DENSENET')
        num_classes, loss_func_test, train_loss, test_loss = transferCheXpertDENSENET(device, train_loader, test_loader,
                                                                            trainData, args.epochs)

        test_labels, pred_cls, pred_proba, _, _ = eval_best_model('densenet121',
                                                                  'checkpoint/densenet121_prechexpert_ftcxr.pth',
                                                                  num_classes, loss_func_test, test_loader, device, band='RGB')
        GradCam(device, 'TransferDensenet')

    elif args.technique == 'SimCLR':
        pretrainSimCLR()
        model_all, loss_func_test = SimCLR(device, train_loader, test_loader, trainData)
        test_labels, pred_cls, pred_proba, _, _ = test_model(model_all, test_loader, loss_func_test)
        GradCam(device, 'SimCLR')


    elif args.technique == 'MoCo':
        pretrainMoCo()
        best_model, loss_func_test = MoCo(device, train_loader, test_loader, trainData)
        test_labels, pred_cls, pred_proba, _, _ = test_model(best_model, test_loader, loss_func_test)
        GradCam(device, 'MoCo')


    #Analysis
    conf_mtrx(test_labels, pred_cls, data)
    target_names = data.classes
    evaluate_metrics(test_labels, pred_cls, target_names)
    ROC_plot_AUC_score(test_labels, pred_proba, len(data.classes), data)
    x_epoch = args.epochs
    plot_learning_curve(x_epoch, train_loss, test_loss)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose specifications.')
    parser.add_argument('--technique', required=True, type=str, metavar='technique', default='baseline RESNET',
                        choices=['baseline DENSENET', 'baseline RESNET', 'transfer IMAGENET RESNET', 'transfer IMAGENET DENSENET', 'transfer CHEXPERT RESNET', 'transfer CHEXPERT DENSENET', 'SimCLR', 'MoCo'],
                        help='Technique')
    parser.add_argument("--RGBPath", help="RGBPath", required=True, type=str, metavar='RGBPath', help="Path to RGB data")
    parser.add_argument("--GREYPath", help="GREYPath", required=True, type=str, metavar='GREYPath', help="Path to greyscale data")
    parser.add_argument('--batchsize', required=True, default=16, type=int, metavar='batchsize',
                        help='Batch size')
    parser.add_argument("--epochs", required=True, default=25 , type=int,  metavar='epochs', help="epochs")
    parser.add_argument("--histogram", required=False, default='false' , type=str,  choices=['true', 'false'],  metavar='histogram', help="histogram")
    parser.add_argument("--showtransformed",  required=False, default='false' , type=str,  choices=['true', 'false'],  metavar='showtransformed',help="showtransformed")

    main(parser.parse_args())






