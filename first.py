import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import os
from typing import List, Tuple


class ImageCorrections:
    """
    Namespace for correction methods applied once to all images
    """

    @staticmethod
    def shades_of_gry(image: np.ndarray, power: int = 6) -> np.ndarray:
        """Applies shades of gray color constancy with using L6 norm of image as a default.
         Requires images with integer color range of 0 to 255
        :param power: norm
        :param image: Matrix: h*w*c
        :return: image with color constancy
        """
        original_type = image.dtype
        new_image = image.astype(np.float)
        if power == np.inf:  # infinity norm
            lightsource = np.max(new_image, (0, 1))
        else:
            new_image = np.power(new_image, power)
            lightsource = np.power(np.mean(new_image, (0, 1)), 1 / power)
        lightsource_normalized = lightsource / np.power(np.sum(np.power(lightsource, 2)), 0.5)
        lightsource_rgb = np.sqrt(3) * lightsource_normalized
        image = np.multiply(image, 1 / lightsource_rgb)
        image = np.clip(image, a_min=0, a_max=255)
        return image.astype(original_type)

    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Applies gamma correction, default value is used in the original shade of gray paper, apply before
        shade_of_gry.
        :param image: image to apply gamma correction on
        :param gamma: gamma correction value
        :return: gamma corrected image
        """
        original_type = image.dtype
        image = image.astype("uint8")
        lookup_table = np.zeros(256, dtype="uint8")
        for i in range(256):
            lookup_table[i] = 255 * np.power((i / 255), 1 / gamma)
        new_image = cv2.LUT(image, lookup_table)
        return new_image.astype(original_type)

    def plot_image(self, image_path: str, tolerance: float = 0.8, gamma: float = 2.2, power: int = 6) -> None:
        """Plots original image and the correction side by side
        :param tolerance: tolerance used for cropping dark edges on image
        :param power: see in function shades_of_gry
        :param gamma: see in function gamma_correction
        :param image_path: path to image
        :return: None, plots images
        """
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        image_cropped = self.crop_circle(image, tolerance)
        image_gamma = self.gamma_correction(image_cropped, gamma)
        image_shades = self.shades_of_gry(image_gamma, power)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(image)
        axs[0, 1].imshow(image_cropped)
        axs[1, 0].imshow(image_gamma)
        axs[1, 1].imshow(image_shades)
        plt.show()

    @staticmethod
    def crop_circle(image: np.ndarray, tolerance: float = 0.8, resize_to_orig: bool = False) -> np.ndarray:
        """
        Crops images with dark edges (for example some images have a circle shape, padded with black (dark) pixels),
        this is unnecessary information, and can hinder training. Cropping is based on the average RGB values in the image.
        :param image: input image
        :param tolerance: bigger the tolerance more cropping happens, unstable performance over 1
        :param resize_to_orig: if cropped image should be resized to original aspect ratio
        :return: cropped image
        """
        mean_all = np.mean(image)
        mean_rows = np.mean(image, (0, 2))
        boundaries = np.where(mean_rows > mean_all * tolerance)
        image_new = image[:, boundaries[0][3]: boundaries[0][-3], :]
        mean_cols = np.mean(image_new, (1, 2))
        boundaries = np.where(mean_cols > mean_all * tolerance)
        image_new = image_new[boundaries[0][3]: boundaries[0][-3], :, :]
        if resize_to_orig:
            image_new = cv2.resize(image_new, (image.shape[1], image.shape[0]))
        return image_new


corrections = ImageCorrections()


# corrections.plot_image("C:/Users/pmarc/PycharmProjects/AI2/melanoma/train/melanoma/ISIC_0000145_downsampled.jpg",
#                        power=6, gamma=2.2)
#
# cropimage = cv2.imread("C:/Users/pmarc/PycharmProjects/AI2/melanoma/train/melanoma/ISIC_0000036_downsampled.jpg")


def apply_corrections(corrections: ImageCorrections, root_path: str) -> None:
    """
    Applies corrections to images in root_path, saves them to root_path + "new"/...
    :param corrections: ImageCorrection class
    :param root_path: path to images
    :return: None
    """
    path_elements = root_path.split("/")
    for idx, element in enumerate(path_elements):
        if element == "":
            path_elements.pop(idx)
    path_modifier = path_elements[-1]
    counter = 0

    for _ in os.walk(root_path):
        counter += 1

    counter2 = 0
    for path, dirs, files in os.walk(root_path):
        print(f"\nApplying to folder {counter2} / {counter}: \n")
        counter2 += 1
        newpath = path.replace(path_modifier, path_modifier + "new")
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with tqdm(total=len(files)) as t:
            for file in files:
                t.update(1)
                filepath = path + "/" + file
                original = cv2.imread(filepath)
                cropped = corrections.crop_circle(original, tolerance=0.8)
                if original.shape[0] * original.shape[1] > 1.1 * cropped.shape[0] * cropped.shape[1]:
                    pass
                else:
                    cropped = original
                gamma_corrected = corrections.gamma_correction(cropped)
                color_corrected = corrections.shades_of_gry(gamma_corrected)
                cv2.imwrite(newpath + "/" + file, color_corrected)


apply_corrections(corrections, "C:/Users/pmarc/PycharmProjects/AI2/melanoma/val/")
apply_corrections(corrections, "C:/Users/pmarc/PycharmProjects/AI2/melanoma/test/")
apply_corrections(corrections, "C:/Users/pmarc/PycharmProjects/AI2/melanoma/train/")
# TODO: rewrite into OOP in the end

device = torch.device('cuda:0')

#  teszt images resized
transform_test = transforms.Compose([transforms.RandomResizedCrop((400, 400), scale=(0.9, 1.0)),
                                     transforms.ToTensor()])

# train images resized, and random transformed for augmentation
transform_train = transforms.Compose([transforms.RandomResizedCrop((400, 400), scale=(0.7, 1.0)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                      transforms.RandomErasing,
                                      transforms.ToTensor()
                                      ])
df_train = datasets.ImageFolder("train", transform=transform_train)
df_test = datasets.ImageFolder("test", transform=transform_test)


# TODO: maybe add some extra channels to pictures besides RGB, with manual functions applied to pics, to get some unique,
# TODO: representations, may make feature extr. easier


def weights(dataset: datasets.ImageFolder) -> Tuple[List[float]]:
    """
    Balancing weights for unbalanced dataset
    :param dataset: set of images with labels
    :return: tuple: list of weights for every individual image, list of weights for every unique label
    """
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

    weights_short = []
    for k, v in counter.items():
        print(len(dataset))
        weights_short.append(len(dataset) / v)
    return weights, weights_short


weights_train_long, weights_train = weights(df_train)

dataloader_train = DataLoader(df_train, batch_size=64, pin_memory=True,
                              shuffle=True)  # , sampler=torch.utils.data.WeightedRandomSampler(weights_train, len(weights_train))

dataloader_test = DataLoader(df_test, batch_size=64, pin_memory=True)


def image_shower(dataloader: DataLoader, width: int = 2, height: int = 2) -> None:
    """
    Shows tensors from iterable data loader as images (width*height number of images).
    :param dataloader: DataLoader object
    :param width: number of image columns on plot
    :param height: number of image rows on plot
    :return: None
    """
    iterator_train = iter(dataloader)
    images, labels = next(iterator_train)
    fig, axes = plt.subplots(figsize=(50, 100), ncols=width, nrows=height)

    idx_to_class_train = {v: k for k, v in df_train.class_to_idx.items()}
    for i in range(width):
        for ii in range(height):
            label = idx_to_class_train[labels[i*width+ii].item()]
            # label = tumor_names[label]
            ax = axes[i, ii]
            image = images[i*width+ii].permute((1, 2, 0))
            ax.imshow(image)
            ax.set_title(label)
    plt.show()


image_shower(dataloader_train)


class CNN(nn.Module):
    """Random neural network, convolutions, max pools, batchnorm etc."""

    # TODO: implement transfer learning with some pretrained models

    def __init__(self):
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
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = torch.squeeze(x)
        x = self.dense(x)
        return x


def run_epoch(data_iterator, model: CNN, optimizer: torch.optim.Optimizer) -> Tuple[np.ndarray]:
    """
    Runs an epoch of training
    :param data_iterator: DataLoader object
    :param model: pytorch model
    :param optimizer: pytorch optimizer
    :return: mean loss and accuracy of the epoch
    """
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
            indiv_loss = nn.functional.cross_entropy(model_out, labels, weight=torch.FloatTensor(weights_train).cuda())
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
def train_model(train_data: DataLoader, test_data: DataLoader, model: CNN, n_epochs: int = 30) -> None:
    """
    Trains neural network.
    :param train_data: training data
    :param test_data: validation data
    :param model: pytorch model
    :param n_epochs: number of epochs
    :return: None
    """
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
