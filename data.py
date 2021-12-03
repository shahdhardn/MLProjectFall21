from PIL import Image
from skimage import io
from skimage.io import imread
import numpy as np
import torch

class CXRDataSet(Dataset):

    def __init__(self, dataset, band, transform=None):
        self.dataset = self.checkchannels(dataset, band)
        self.transform = transform

    def checkchannels(self, dataset, band):

        if band == 'RGB':
            datasetRGB = []
            for index in range(len(dataset)):
                if Image.open(dataset[index][0]).getbands() == ('R', 'G', 'B'):
                    datasetRGB.append(dataset[index])
            return datasetRGB
        elif band == 'L':
            datasetG = []
            for index in range(len(dataset)):
                if Image.open(dataset[index][0]).getbands() == ('L',):
                    datasetG.append(dataset[index])
            return datasetG

    def croppingframe(self, index):  # Custom transform
        image = io.imread(self.dataset[index][0])
        nonzero_pixels = image[np.nonzero(image)]
        if nonzero_pixels.shape == (0,):
            return image
        min = np.array(np.nonzero(image)).min(axis=1)
        max = np.array(np.nonzero(image)).max(axis=1)
        return image[min[0]:max[0] + 1, min[1]:max[1] + 1]

    def __getitem__(self, index):  # allows us to index our instance

        image = self.croppingframe(index)
        image = Image.fromarray(image)
        y_label = torch.tensor(self.dataset[index][1])
        if self.transform:
            return self.transform(image), y_label

        return image, y_label

    def __len__(self):
        return len(self.dataset)
