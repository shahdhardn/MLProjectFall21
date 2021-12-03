import os
import torch

from test import test_model


def train_nn(model, train_loader, test_loader, loss_func_train, loss_func_test, optimizer, epochs, modelname, device):
    os.makedirs('checkpoint', exist_ok=True)
    best_val_acc = 0
    train_loss = []
    test_loss = []
    checkpoint_path = f'{modelname}.pth'

    for epoch in range(epochs):

        print(f'Epoch number {epoch}')
        # set the model to training mode:
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for batch in train_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)  # get integer value

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = loss_func_train(outputs, labels)
            # label is a single value, output is a vector
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (running_correct / total) * 100

        train_loss.append(epoch_loss)
        print(f'Train: {epoch_acc}% of the images classified correctly. Epoch loss= {epoch_loss}')

        test_labels, pred_cls, pred_proba, t_loss, epoch_val_acc = test_model(model, test_loader, loss_func_test, device)

        test_loss.append(t_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            #  save_checkpoint(model, optimizer,epoch,best_val_acc, modelname)
            torch.save(model.state_dict(), 'checkpoint' + '/' + checkpoint_path)

    print('Done with training and testing..')
    return model, test_labels, pred_cls, pred_proba, train_loss, test_loss