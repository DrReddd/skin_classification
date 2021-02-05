import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO: rewrite into OOP in the end

device = torch.device('cuda:0')

#  teszt images resized
transform_test = transforms.Compose([transforms.Resize((260, 160)),
                                     transforms.ToTensor()])

# train images resized, and random transformed for augmentation
transform_train = transforms.Compose([transforms.Resize((260, 160)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                      transforms.ToTensor()
                                      ])
df_train = datasets.ImageFolder("train", transform=transform_train)
df_test = datasets.ImageFolder("test", transform=transform_test)

# TODO: maybe add some extra channels to pictures besides RGB, with manual functions applied to pics, to get some unique,
# TODO: representations, may make feature extr. easier


def weights(dataset):
    """Balancing weights for unbalanced dataset"""
    counter = {}
    weights = []
    sumup = len(dataset.imgs)
    for image in dataset.imgs:
        try:
            counter[image[1]] += 1
        except KeyError:
            counter[image[1]] = 0
    for image in dataset.imgs:
        weights.append(sumup / counter[image[1]])
    return weights


weights_train = weights(df_train)


def dic_invert(dic):
    "Invert a dictionary, only use with 1-1 key-value dics"
    return {v: k for k, v in dic.items()}


#  training data unbalanced sample using weights
dataloader_train = DataLoader(df_train, sampler=torch.utils.data.WeightedRandomSampler(
    weights_train, len(weights_train)), batch_size=64, pin_memory=True)

dataloader_test = DataLoader(df_test, batch_size=64, pin_memory=True)


def image_shower(dataloader):
    """Shows tensors from iterable data loader as images, hardcoded for 4 image as of now"""
    iterator_train = iter(dataloader)
    images, labels = next(iterator_train)
    fig, axes = plt.subplots(figsize=(50, 100), ncols=4)

    idx_to_class_train = dic_invert(df_train.class_to_idx)
    for i in range(4):
        label = idx_to_class_train[labels[i].item()]
        #label = tumor_names[label]
        ax = axes[i]
        image = images[i].permute((1, 2, 0))
        ax.imshow(image)
        ax.set_title(label)
    plt.show()

    return None


image_shower(dataloader_train)


class CNN(nn.Module):
    """Random neural network, convolutions, max pools, batchnorm etc."""
    # TODO: implement transfer learning with some pretrained models

    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(128),
            nn.AvgPool2d((62, 37)),  # fuck knows, debug

        )
        self.dense = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.squeeze(x)
        x = self.dense(x)
        return x


def run_epoch(data_iterator, model, optimizer):
    """Runs an epoch"""
    loss = []
    acc = []
    with tqdm(total=len(data_iterator)) as t:
        for idx, data in enumerate(data_iterator):
            t.update(1)
            labels = data[1].cuda()
            data = data[0].cuda()
            if model.training is False:
                with torch.no_grad():
                    model_out = model.forward(data)
            else:
                model_out = model.forward(data)
            indiv_loss = nn.functional.cross_entropy(model_out, labels)
            loss.append(indiv_loss.item())
            prediction = torch.argmax(model_out, dim=1)
            asd = np.equal(prediction.cpu().numpy(), labels.cpu().numpy())
            accuracy = np.mean(asd)
            acc.append(accuracy)
            if model.training is True:
                optimizer.zero_grad()
                indiv_loss.backward()
                optimizer.step()

    return np.mean(loss), np.mean(acc)


# extra params for optional SGD:
def train_model(train_data, test_data, model, lr=0.1, momentum=0, nesterov=False, n_epochs=30):
    """Train a model for N epochs given data and hyper-params."""
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, n_epochs + 1):
        print(f"-------------\nEpoch {epoch} / {n_epochs}:\n")

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print(f"Loss: {loss:.6f}, Accuracy: {acc:.6f}")

        # Run **validation** although something have to be done about unbalanced stuff
        # TODO: rethink validation, add additional fit indecies
        val_loss, val_acc = run_epoch(test_data, model.eval(), optimizer)
        print(f"Validation loss: {val_loss:.6f}, Validation accuracy: {val_acc:.6f}")

    # Save model
    torch.save(model, 'model420.pt')


model = CNN(None).to(device)
train_model(dataloader_train, dataloader_test, model)
