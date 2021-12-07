import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly

num_workers = 8
input_size = 224
batch_size = 32
memory_bank_size = 2048
seed = 1
max_epochs = 20


class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]



def pretrainMoCo():
    path_to_data = '/home/eman.alsuradi/Desktop/ml701 project/Chexpert_SSL'
    pl.seed_everything(seed)

    collate_fn = lightly.data.MoCoCollateFunction(
        input_size=input_size,
        vf_prob=0.5,
        hf_prob=0.5,
        rr_prob=0.5,
        gaussian_blur=0.0,
        cj_prob=0.0,
        random_gray_scale=0.0
    )

    dataset_train_moco = lightly.data.LightlyDataset(
        input_dir=path_to_data)

    dataloader_train_moco = torch.utils.data.DataLoader(
        dataset_train_moco,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers)

    # use a GPU if available
    gpus = 1 if torch.cuda.is_available() else 0

    model = MocoModel()
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                         progress_bar_refresh_rate=100)
    trainer.fit(
        model,
        dataloader_train_moco
    )

    # You could use the pre-trained model and train a classifier on top.
    pretrained_resnet_moco = model

    # you can also store the backbone and use it in another code
    state_dict = {
        'resnet18_parameters': pretrained_resnet_moco.state_dict()
    }
    torch.save(state_dict, 'model_moco.pth')
