import numpy as np
from glob import glob
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from data import CXRDataSet, calculate_cls_weight
from train import train_nn


def prepare_dataset(dataframe, class_names):
    dataset_df = dataframe[dataframe['Frontal/Lateral'] == 'Frontal']  # take frontal pics only
    df = dataset_df.sample(frac=1., random_state=1)
    df.fillna(0, inplace=True)  # fill the with zeros
    x_path, y_df = df["full_path"].to_numpy(), df[class_names]
    y = np.empty(y_df.shape, dtype=int)
    for i, (index, row) in enumerate(y_df.iterrows()):
        labels = []
        for cls in class_names:
            curr_val = row[cls]
            feat_val = 0
            if curr_val:
                curr_val = float(curr_val)
                if curr_val == 1:
                    feat_val = 1
                elif curr_val == -1:
                    feat_val = 0
            else:
                feat_val = 0

            labels.append(feat_val)

        y[i] = labels

    return x_path, y



def preprocess_train(device, network):
    # It will collect the paths of all images and store them in my_glob
    my_glob_train = glob('CheXpert-v1.0-small/train/patient*/study*/*.jpg')
    print('Before preprocessing CheXpert dataset, number of train images: ', len(my_glob_train))

    my_glob_valid = glob('CheXpert-v1.0-small/valid/patient*/study*/*.jpg')
    print('Before preprocessing CheXpert dataset, number of validation images: ', len(my_glob_valid))

    train_df = pd.read_csv('CheXpert-v1.0-small/train.csv')
    print(f'Before preprocessing CheXpert dataset,the shape of the training dataset is : {train_df.shape}')

    valid_df = pd.read_csv('CheXpert-v1.0-small/valid.csv')
    print(f'Before preprocessing CheXpert dataset,the shape of the training dataset is : {valid_df.shape}')

    full_img_paths = {'CheXpert-v1.0-small/' + x[18:]: x for x in my_glob_train}
    train_df['full_path'] = train_df['Path'].map(full_img_paths.get)

    full_img_paths = {'CheXpert-v1.0-small/' + x[18:]: x for x in my_glob_valid}
    valid_df['full_path'] = valid_df['Path'].map(full_img_paths.get)

    class_names = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    x_path_train, labels_train = prepare_dataset(train_df, class_names)
    x_path_valid, labels_valid = prepare_dataset(valid_df, class_names)

    # Take all paths of images and create a dataframe from them
    train_df = pd.DataFrame({'path': x_path_train})
    # Create a new dataframe for the labels. Here the labels are in the form of one-hot vector
    df2_train = pd.DataFrame(labels_train, columns=class_names)
    # Add the second dataframe to the first one
    train_df[list(df2_train.columns)] = df2_train

    # Valid
    valid_df = pd.DataFrame({'path': x_path_valid})
    df2_valid = pd.DataFrame(labels_valid, columns=class_names)
    valid_df[list(df2_valid.columns)] = df2_valid

    df_prepare_train = train_df[df2_train.sum(axis=1) == 1]
    df_prepare_train = df_prepare_train.reset_index(drop=True)

    df_prepare_valid = valid_df[df2_valid.sum(axis=1) == 1]
    df_prepare_valid = df_prepare_valid.reset_index(drop=True)

    # Convert the one-hot vectors into categorical classes
    # 'No Finding': 0 , 'Atelectasis':1, 'Cardiomegaly':2, 'Consolidation':3 , 'Edema':4 , 'Pleural Effusion':5
    final_labels_train = []
    for (index, row) in (df_prepare_train.iterrows()):
        for cls in class_names:
            if row[cls] == 1:
                final_labels_train.append(class_names.index(cls))

    final_labels_valid = []
    for (index, row) in (df_prepare_valid.iterrows()):
        for cls in class_names:
            if row[cls] == 1:
                final_labels_valid.append(class_names.index(cls))

    images_list_train = df_prepare_train['path'].tolist()
    TrainData_chexpert = list(zip(images_list_train, final_labels_train))

    images_list_valid = df_prepare_valid['path'].tolist()
    TestData_chexpert = list(zip(images_list_valid, final_labels_valid))

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

    train_set_chexpert = CXRDataSet(TrainData_chexpert, 'L',
                                    transform=train_data_transforms)
    test_set_chexpert = CXRDataSet(TestData_chexpert, 'L',
                                   transform=test_data_transforms)

    print("Size of train dataset: ", len(train_set_chexpert))
    print("Size of test dataset: ", len(test_set_chexpert))

    batch_size = 32  # Fixed value

    train_loader_chexpert = DataLoader(dataset=train_set_chexpert,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8)
    test_loader_chexpert = DataLoader(dataset=test_set_chexpert,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=8)

    ## Start pertraining process
    if network=='RESNET':
        # preprocess_train function(model) model = 'resnet'
        res_chxtrans = models.resnet18(pretrained=False)
        res_chxtrans.inplanes = 64
        res_chxtrans.conv1 = nn.Conv2d(1, res_chxtrans.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        input_ftrs = res_chxtrans.fc.in_features
        num_classes = 6
        res_chxtrans.fc = nn.Linear(input_ftrs, num_classes)

        # transfer the model to gpu
        res_chxtrans = res_chxtrans.to(device)

        # #instantiate crossentropy loss object
        loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(TrainData_chexpert))
        loss_func_test = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(res_chxtrans.parameters(), lr=0.001)

        modelname = 'resnet18_prechexpert'
        epochs = 100  ## Fixed Value
        model, _, _, _, _, _ = train_nn(res_chxtrans,
                                        train_loader_chexpert, test_loader_chexpert,
                                        loss_func_train, loss_func_test, optimizer, epochs, modelname)

    elif network=='DENSENET':
        # preprocess_train function (model) model =densenet
        dens_chxtrans = models.densenet121(pretrained=False)
        num_init_features = 64
        dens_chxtrans.features.conv0 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                                 padding=3, bias=False)

        num_ftrs = dens_chxtrans.classifier.in_features
        num_classes = 6
        dens_chxtrans.classifier = nn.Linear(num_ftrs, num_classes)

        dens_chxtrans = dens_chxtrans.to(device)

        loss_func_test = nn.CrossEntropyLoss()
        loss_func_train = nn.CrossEntropyLoss(weight=calculate_cls_weight(TrainData_chexpert))

        optimizer = optim.AdamW(dens_chxtrans.parameters(), lr=0.001)

        modelname = 'densenet121_prechexpert'

        epochs = 100  # fixed value will not be

        model, _, _, _, _, _ = train_nn(dens_chxtrans,
                                        train_loader_chexpert, test_loader_chexpert,
                                        loss_func_train, loss_func_test, optimizer, epochs, modelname)








