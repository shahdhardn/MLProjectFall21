import torchvision.models as models
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from test import test_model


def eval_best_model(model, model_path, n_classes, loss_func_test, test_loader, device, band='RGB'):
    if model == 'resnet18':
        best_model = models.resnet18()
        input_ftrs = best_model.fc.in_features
        best_model.fc = nn.Linear(input_ftrs, n_classes)
        if band == 'L':
            best_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif model == 'densenet121':
        best_model = models.densenet121()
        input_ftrs = best_model.classifier.in_features
        best_model.classifier = nn.Linear(input_ftrs, n_classes)
        if band == 'L':
            best_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    ckpt = torch.load(model_path)
    best_model.load_state_dict(ckpt)
    best_model = best_model.to(device)
    test_labels, pred_cls, pred_proba, t_loss, test_acc = test_model(best_model, test_loader, loss_func_test)
    return test_labels, pred_cls, pred_proba, t_loss, test_acc


def conf_mtrx(test_labels, pred_cls, data):
    test_labels = torch.cat(test_labels).to('cpu')  # cat= concatenate
    pred_cls = torch.cat(pred_cls).to('cpu')

    conf_matrx = confusion_matrix(test_labels, pred_cls)

    class_names = data.classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(conf_matrx), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    # You can change the position of the title by providing
    # a value for the y parameter
    plt.title('Confusion matrix [0:Atypical, 1:Indeterminate, 2:Negative, 3:Typical]', y=1.2, fontsize=14,
              fontweight='bold')
    plt.ylabel('Actual Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')


def ROC_plot_AUC_score(test_labels, pred_proba, n_classes, data):
    test_labels = torch.cat(test_labels).to('cpu')  # cat= concatenate
    pred_proba = torch.cat(pred_proba, dim=0).to('cpu')

    soft_func = nn.Softmax(dim=1)
    pred_proba = soft_func(pred_proba)

    fpr = {}  # False postive rate x-axis
    tpr = {}  # True positive rate y-axis
    thresh = {}
    auc_scores = []
    one_vs_all_labels = []

    for i in range(n_classes):
        one_vs_all_labels.append((test_labels == i).numpy().astype('int'))

    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(one_vs_all_labels[i], pred_proba[:, i], pos_label=1)
        auc_scores.append(roc_auc_score(one_vs_all_labels[i], pred_proba[:, i]))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], linestyle='--', label=f'Class {data.classes[i]} vs Rest')

    plt.title('Multiclass ROC Curve', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.legend(loc='best')

    # Display AUC scores

    for i in range(n_classes):
        print(f'AUC score of Class {data.classes[i]} vs Rest ===>', auc_scores[i])


def evaluate_metrics(test_labels, pred_cls, target_names):
    test_labels = torch.cat(test_labels).to('cpu')  # cat= concatenate
    pred_cls = torch.cat(pred_cls).to('cpu')

    print(classification_report(test_labels, pred_cls, target_names=target_names))


def plot_learning_curve(x_epoch, train_loss, test_loss):
    print('test_loss:', test_loss)
    x_epoch = np.arange(x_epoch)
    plt.plot(x_epoch, train_loss, linestyle='--', label='Train loss')
    plt.plot(x_epoch, test_loss, linestyle='--', label='Test loss')

    plt.title('Learning Curve- loss against epochs plot', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.show()
