import torch

def test_model(model, test_loader, loss_func_test, device):
    model.eval()
    predicted_correctly = 0
    total = 0
    test_running_loss = 0
    t_loss = 0.0
    # Variables defined for post-training analysis
    test_labels = []
    pred_cls = []
    pred_proba = []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)  # get integer value

            test_labels.append(labels)
            outputs = model(images)
            pred_proba.append(outputs)

            loss = loss_func_test(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            pred_cls.append(predicted)

            predicted_correctly += (predicted == labels).sum().item()
            test_running_loss += loss.item() * labels.size(0)

    test_acc = (predicted_correctly / total) * 100
    t_loss = test_running_loss / total

    print(f'Test: {test_acc}% of the images are classified correctly. Epoch loss = {t_loss}')

    return test_labels, pred_cls, pred_proba, t_loss, test_acc
