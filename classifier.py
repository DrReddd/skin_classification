import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import List, Tuple, Optional
from efficientnet_pytorch import EfficientNet
import efficientnet_pytorch
import pandas as pd
import json
import metadata_functions


class ImageCorrections:
    """
    Namespace for correction methods applied once to all images
    """

    @staticmethod
    def shades_of_gry(image: np.ndarray, power: int = 6) -> np.ndarray:
        """
        Applies shades of gray color constancy with using L6 norm of image as a default.
        Requires images with integer color range of 0 to 255

        Parameters
        ----------
        image: np.ndarray
            Image to apply correction on
        power: int
            Which norm to use for correction

        Returns
        -------
        np.ndarray
            Corrected image

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
        """
        Applies gamma correction, default value is used in the original shade of gray paper, apply before
        shade_of_gry.

        Parameters
        ----------
        image:
            Image to apply gamma correction on
        gamma:
            Gamma correction value

        Returns
        -------
        np.ndarray
            Gamma corrected image
        """
        original_type = image.dtype
        image = image.astype("uint8")
        lookup_table = np.zeros(256, dtype="uint8")
        for i in range(256):
            lookup_table[i] = 255 * np.power((i / 255), 1 / gamma)
        new_image = lookup_table[image]  # cv2.LUT(image, lookup_table)
        return new_image.astype(original_type)

    @staticmethod
    def crop_circle(image: np.ndarray, tolerance: float = 0.8, resize_to_orig: bool = False) -> np.ndarray:
        """
        Crops images with dark edges (for example some images have a circle shape, padded with black (dark) pixels),
        this is unnecessary information, and can hinder training.
        Cropping is based on the average RGB values in the image.

        Parameters
        ----------
        image:
            Input image
        tolerance:
            Bigger the tolerance more cropping happens, unstable performance over 1
        resize_to_orig:
            If cropped image should be resized to original aspect ratio

        Returns
        -------
        np.ndarray
            Cropped image
        """
        mean_all = np.mean(image)
        mean_rows = np.mean(image, (0, 2))
        boundaries = np.where(mean_rows > mean_all * tolerance)
        image_new = image[:, boundaries[0][3]: boundaries[0][-3], :]
        mean_cols = np.mean(image_new, (1, 2))
        boundaries = np.where(mean_cols > mean_all * tolerance)
        image_new = image_new[boundaries[0][3]: boundaries[0][-3], :, :]
        if resize_to_orig:
            image_new = resize(image_new, (image.shape[0], image.shape[1]))
        return image_new

    def plot_image(self, image_path: str, tolerance: float = 0.8, gamma: float = 2.2, power: int = 6) -> None:
        """
        Plots original image and the correction side by side

        Parameters
        ----------
        image_path:
            Path to image file
        tolerance:
            Tolerance used for cropping dark edges on image
        gamma:
            Gamma correction value
        power:
            Which norm to use for shades of gray correction

        Returns
        -------
        None

        """
        image = plt.imread(image_path)
        image_cropped = self.crop_circle(image, tolerance)
        image_gamma = self.gamma_correction(image_cropped, gamma)
        image_shades = self.shades_of_gry(image_gamma, power)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(image)
        axs[0, 1].imshow(image_cropped)
        axs[1, 0].imshow(image_gamma)
        axs[1, 1].imshow(image_shades)
        plt.show()


class CNN(nn.Module):
    """
    A simple CNN architecture
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, (12, 12)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(),
            nn.AvgPool2d((16, 16)),  # fuck knows, debug

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
        return x[None, ...]


class EfficientnetMetadata(nn.Module):
    """
    Model made by adding a metadata layer to a pretrained CNN model
    """
    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.pretrained = pretrained_model
        self.connect_meta = nn.Linear(50 + 13, 90)
        self.final = nn.Linear(90, 3)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        x1 = self.pretrained(image)
        x2 = metadata
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.connect_meta(x))
        x = self.final(x)
        return x


class Metadata(nn.Module):
    """
    A very simple model using only metadata, without any image data
    """
    def __init__(self):
        super().__init__()
        self.connect_meta = nn.Linear(13, 40)
        self.final = nn.Linear(40, 3)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        x = metadata
        x = F.relu(self.connect_meta(x))
        x = self.final(x)
        return x


class Hooks:
    """
    Class for hooking model modules, collecting and displaying internal data during back or forward propagation

    Arguments
    ---------
    model: nn.Module
        pytorch model
    which_cnn_layer: int
        which convolution layer to hook, default is first (0)

    Attributes
    ----------
    gradientlist: List[torch.Tensor]
        for storing gradient vectors or weights during back or forward propagation for output of a hooked module
    gradientlist_in: List[torch.Tensor]
        for storing gradient vectors or weights during back or forward propagation for input of a hooked module

    """

    def __init__(self, model: nn.Module, which_cnn_layer: int = 0):
        self.model = model
        self.which_cnn_layer = which_cnn_layer
        self.gradientlist = []
        self.gradientlist_in = []
        self.model.eval()
        self.hooker()

    def hooker(self) -> None:
        """
        Registers hooks defined in inner functions.

        Returns
        -------
        None

        """
        def backw_hook_cnn(module: nn.Module, grad_input: Tuple, grad_output: Tuple):
            """
            Backwards hook

            Parameters
            ----------
            module:
                module to hook
            grad_input:
                input gradient
            grad_output:
                output gradient

            Returns
            -------
            None

            """
            self.gradientlist_in = grad_input
            for input in grad_input:
                if input is not None:
                    print(input.shape)
            self.gradientlist = []
            output = grad_output[0].squeeze().cpu().numpy()
            for i in range(output.shape[0]):
                output_abs = np.abs(output[i, ...])
                output_element = output[i, ...]
                self.gradientlist.append(output_element)

        def forw_hook_cnn(module: nn.Module, input: Tuple, output: torch.Tensor):
            """
            Forwards hook

            Parameters
            ----------
            module:
                module to hook
            input:
                input
            output:
                output

            Returns
            -------
            None

            """
            self.gradientlist = []
            self.gradientlist_in = module.weight.cpu().detach().numpy()
            output = output.squeeze().cpu().detach().numpy()
            for i in range(output.shape[0]):
                output_abs = np.abs(output[i, ...])
                output_element = output[i, ...]
                self.gradientlist.append(output_element)

        def guided_swish_hook(module: nn.Module, grad_in: Tuple[torch.Tensor, ...],
                              grad_out: Tuple[torch.Tensor, ...]) -> Tuple:
            return (torch.clamp(grad_in[0], min=0.0),)

        conv_layer_counter = 0
        for _, module in self.model.named_modules():

            if isinstance(module, nn.modules.ReLU):
                module.register_backward_hook(guided_swish_hook)
            elif isinstance(module, efficientnet_pytorch.utils.MemoryEfficientSwish):
                module.register_backward_hook(guided_swish_hook)
            elif isinstance(module, nn.modules.conv.Conv2d):
                if conv_layer_counter == self.which_cnn_layer:
                    module.register_backward_hook(backw_hook_cnn)
                    module.register_forward_hook(forw_hook_cnn)
                conv_layer_counter += 1

    def saliency(self, image: np.ndarray, label: torch.Tensor, metadata: torch.Tensor = None) -> None:
        """
        Visualizes image activation based on forward prop and backward prop using the gradients from
        first convolution layer.

        Parameters
        ----------
        image:
            Image to visualize
        label:
            Label of the image for loss function
        metadata:
            Possible metadata input for appropriate models

        Returns
        -------
        None

        """
        transform_test1 = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop((300, 300), scale=(0.7, 1.0))])
        transform_test2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.796, 0.784, 0.778], [0.0904, 0.148, 0.124])])
        input_mid = transform_test1(image)
        input = transform_test2(input_mid)
        input.requires_grad = True
        if metadata is None:
            model_out = self.model(input[None, ...].to(device))
        else:
            model_out = self.model(input[None, ...].to(device), metadata[None, ...].to(device))
        indiv_loss = nn.functional.cross_entropy(model_out, label.to(device),
                                                 weight=torch.tensor(weights_train).to(device))

        # cnn weights, and convolution result plotted:
        fig, axs = plt.subplots(5, 8)
        # self.gradientlist_in = np.interp(self.gradientlist_in, (self.gradientlist_in.min(), self.gradientlist_in.max()), (0, 1))
        for i in range(5):
            for ii in range(8):
                if i == 0 and ii == 0:
                    axs[i, ii].imshow(np.array(input_mid))
                else:
                    asd = self.gradientlist_in[5 * i + ii - 1].squeeze()
                    # asd = asd*np.array([0.0904, 0.148, 0.124])[:, None, None]+np.array([0.796, 0.784, 0.778])[:, None, None]
                    asd = asd.transpose((1, 2, 0))
                    asd = np.interp(asd, (asd.min(), asd.max()), (0, 1))
                    axs[i, ii].imshow(asd)
        fig, axs = plt.subplots(5, 8)
        for i in range(5):
            for ii in range(8):
                if i == 0 and ii == 0:
                    axs[i, ii].imshow(np.array(input_mid))
                else:
                    axs[i, ii].imshow(self.gradientlist[5 * i + ii - 1], cmap="seismic")

        plt.show()

        self.model.zero_grad()
        indiv_loss.backward()

        # basic guided saliency map:
        saliency_input = self.gradientlist_in[0].squeeze()
        saliency_input = saliency_input.cpu().numpy().transpose((1, 2, 0))
        saliency_input[np.where(saliency_input < 0)] = 0
        saliency_input = np.interp(saliency_input, (saliency_input.min(), saliency_input.max()), (0, 1))
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(saliency_input)
        axs[1].imshow(np.array(input_mid))
        plt.show()

        # output gradients of first cnn:
        fig, axs = plt.subplots(5, 8)
        for i in range(5):
            for ii in range(8):
                if i == 0 and ii == 0:
                    axs[i, ii].imshow(np.array(input_mid))
                else:
                    axs[i, ii].imshow(self.gradientlist[5 * i + ii - 1], cmap="magma")

        plt.show()


def apply_corrections(corrections: ImageCorrections, root_path: str) -> None:
    """
    Applies corrections to images in root_path, saves them to root_path + "new"/...

    Parameters
    ----------
    corrections:
        ImageCorrection class containing functions to apply to all images
    root_path:
        path to images

    Returns
    -------
    None

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
                original = plt.imread(filepath)
                cropped = corrections.crop_circle(original, tolerance=0.8)
                if original.shape[0] * original.shape[1] > 1.1 * cropped.shape[0] * cropped.shape[1]:
                    pass
                else:
                    cropped = original
                gamma_corrected = corrections.gamma_correction(cropped)
                color_corrected = corrections.shades_of_gry(gamma_corrected)
                plt.imsave(newpath + "/" + file, color_corrected)


# apply_corrections(corrections, "C:/Users/pmarc/PycharmProjects/AI2/melanoma/val/")
# apply_corrections(corrections, "C:/Users/pmarc/PycharmProjects/AI2/melanoma/test/")
# apply_corrections(corrections, "C:/Users/pmarc/PycharmProjects/AI2/melanoma/train/")


def weights(dataset: metadata_functions.ImageFolderMetadata) -> Tuple[List[float], List[float]]:
    """
    Computes balancing weights for unbalanced dataset

    Parameters
    ----------
    dataset:
        Dataset of images with labels.

    Returns
    -------
    tuple: list of weights for every individual image, list of weights for every unique label

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
        weights_short.append(len(dataset) / v)
    return weights, weights_short


def image_shower(dataloader: DataLoader, width: int = 2, height: int = 2) -> None:
    """
    Shows tensors from iterable data loader as images (width*height number of images).

    Parameters
    ----------
    dataloader:
        Dataloader object to load images, labels
    width:
        Number of image columns on plot
    height:
        Number of image rows on plot

    Returns
    -------
    None

    """
    iterator_train = iter(dataloader)
    images_labels = next(iterator_train)
    images = images_labels[0]
    labels = images_labels[1]
    fig, axes = plt.subplots(figsize=(50, 100), ncols=width, nrows=height)
    idx_to_class_train = {v: k for k, v in df_train.class_to_idx.items()}
    for i in range(width):
        for ii in range(height):
            label = idx_to_class_train[labels[i * width + ii].item()]
            # label = tumor_names[label]
            ax = axes[i, ii]
            image = images[i * width + ii].permute((1, 2, 0))
            ax.imshow(image)
            ax.set_title(label)
    plt.show()


def mean_sd(data_iterator: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates mean and standard deviation values of RGB channels globally across images in data_iterator generator.

    Parameters
    ----------
    data_iterator:
        DataLoader object for generating the data.

    Returns
    -------
    Mean and sd matrices in torch format

    """
    mean = torch.zeros(3)
    sd = torch.zeros(3)

    with tqdm(total=len(data_iterator)) as t:
        for idx, data in enumerate(data_iterator):
            t.update(1)
            image = data[0]
            mean += torch.mean(image, (0, 2, 3))
            sd += torch.std(image, (0, 2, 3))
    mean = mean / len(data_iterator)
    sd = sd / len(data_iterator)
    return mean, sd


def run_epoch(data_iterator: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer = None,
              is_test: bool = False, is_metadata: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Runs an epoch of training (or testing)

    Parameters
    ----------
    data_iterator:
        DataLoader object
    model:
        Pytorch model
    optimizer:
        Pytorch optimizer
    is_test:
        Set true if the function is used in testing the model with test set, so it returns confusion matrix too.
    is_metadata:
        Is there additional metadata besides images and labels (for model using metadata)

    Returns
    -------
    Mean loss and accuracy of the epoch, and optionally confusion matrix

    """
    loss = []
    acc = []
    confusion_m = torch.zeros((3, 3))
    with tqdm(total=len(data_iterator)) as t:
        for idx, data in enumerate(data_iterator):
            t.update(1)
            labels = data[1].to(device)
            image = data[0].to(device)
            if is_metadata:
                metadata = data[2].to(device)
                if not model.training:
                    with torch.no_grad():
                        model_out = model.forward(image, metadata)
                else:
                    model_out = model.forward(image, metadata)
            else:
                if not model.training:
                    with torch.no_grad():
                        model_out = model.forward(image)
                else:
                    model_out = model.forward(image)

            indiv_loss = nn.functional.cross_entropy(model_out, labels,
                                                     weight=torch.FloatTensor(weights_train).to(device))
            loss.append(indiv_loss.item())
            prediction = torch.argmax(model_out, dim=1)
            asd = np.equal(prediction.cpu().numpy(), labels.cpu().numpy())
            accuracy = np.mean(asd)
            acc.append(accuracy)
            if is_test is True:
                for idx2, label in enumerate(labels):
                    confusion_m[label.item(), prediction[idx2].item()] += 1
            if model.training is True:
                optimizer.zero_grad()
                indiv_loss.backward()
                optimizer.step()
    if is_test is True:
        return np.mean(loss), np.mean(acc), confusion_m.numpy()
    return np.mean(loss), np.mean(acc), None


# extra params for optional SGD:
def train_model(train_data: DataLoader, test_data: DataLoader, model: nn.Module, n_epochs: int = 50,
                is_metadata: bool = False) -> None:
    """

    Parameters
    ----------
    train_data:
        Training data (DataLoader object)
    test_data:
        Validation data (DataLoader object)
    model:
        Pytorch model
    n_epochs:
        Number of epochs to train
    is_metadata:
        Is there additional metadata besides images and labels (for model using metadata)

    Returns
    -------
    None

    """
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, n_epochs + 1):
        print(f"-------------\nEpoch {epoch} / {n_epochs}:\n")

        # Run **training***
        loss, acc, _ = run_epoch(train_data, model.train(), optimizer, is_metadata=is_metadata)
        print(f"Loss: {loss:.6f}, Accuracy: {acc:.6f}")

        # Run **validation**
        val_loss, val_acc, _ = run_epoch(test_data, model.eval(), optimizer, is_metadata=is_metadata)
        print(f"Validation loss: {val_loss:.6f}, Validation accuracy: {val_acc:.6f}")

        # Save model
        torch.save(model, f"models/model421_iteration{epoch}.pt")
    torch.save(model, f"models/model421_final.pt")


def test_model(test_data: DataLoader, model: nn.Module, is_metadata: bool = False) -> None:
    """
    Tests neural network

    Parameters
    ----------
    test_data:
        Test data (DataLader object)
    model:
        Pytorch model
    is_metadata:
        Is there additional metadata besides images and labels (for model using metadata)

    Returns
    -------
    None

    """
    optimizer = torch.optim.Adam(model.parameters())

    # Run **testing**
    test_loss, test_acc, confusion_m = run_epoch(test_data, model.eval(), optimizer, is_test=True,
                                                 is_metadata=is_metadata)
    print(f"Test loss: {test_loss:.6f}, Test accuracy: {test_acc:.6f}, Confusion matrix:\n {confusion_m}")


def test_images(images: List[np.ndarray], model: nn.Module, labels: Optional[List[int]] = None,
                metadata: Optional[List[torch.Tensor]] = None) -> None:
    """
    Classifies a list of images based on a given model, shows the pictures, and probabilites, and optionally
    associated labels (ground truth).

    Parameters
    ----------
    images:
        List of images to test
    model:
        Pytorch model
    labels:
        List of associated labels
    metadata:
        List of associated metadata

    Returns
    -------
    None

    """
    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.RandomResizedCrop((300, 300), scale=(0.7, 1.0)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.796, 0.784, 0.778], [0.0904, 0.148, 0.124])])
    list_of_probs = []
    for idx, image in enumerate(images):
        probabilities = np.zeros(3)
        for _ in range(5):
            input = transform_test(image)
            with torch.no_grad():
                if metadata is None:
                    model_out = model(input[None, ...].to(device))
                else:
                    model_out = model(input[None, ...].to(device), metadata[idx][None, ...].to(device))
                softm = nn.Softmax(dim=1)
                model_out = softm(model_out)
            model_out = model_out.cpu().numpy()
            probabilities += model_out.reshape(3)
        probabilities /= 5
        list_of_probs.append(probabilities)
        plt.imshow(image, aspect="equal")
        if labels is not None:
            plt.title(f"Probablities: Melanoma with {probabilities[0]:.3f} ({labels[idx] == 0}),\n "
                      f"Naevus with {probabilities[1]:.3f} ({labels[idx] == 1}),\n"
                      f"Other with {probabilities[2]:.3f} ({labels[idx] == 2})"
                      )
        else:
            plt.title(f"Probablities: Melanoma with {probabilities[0]:.3f}, Naevus with {probabilities[1]:.3f},"
                      f"Other with {probabilities[2]:.3f}")
        plt.show()


if __name__ == '__main__':

    device = torch.device("cuda:0")
    print("Gimme input (train for training, test for testing, imgs for some images to classify):\n")
    what = input()
    #  preprocessing mean and sd
    transform_pre = transforms.Compose([transforms.RandomResizedCrop((300, 300), scale=(0.9, 1.0)),
                                        transforms.ToTensor()])

    #  teszt images resized
    transform_test = transforms.Compose([transforms.RandomResizedCrop((300, 300), scale=(0.9, 1.0)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.796, 0.784, 0.778], [0.0904, 0.148, 0.124])])

    # train images resized, and random transformed for augmentation
    transform_train = transforms.Compose([transforms.RandomResizedCrop((300, 300), scale=(0.7, 1.0)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.796, 0.784, 0.778], [0.0904, 0.148, 0.124]),
                                          transforms.RandomErasing(scale=(0.02, 0.2))
                                          ])

    data = pd.read_csv("metadata/ISIC_2019_Training_Metadata.csv")
    metadata_functions.jsonify(data["image"].tolist(), data, ["age_approx", "anatom_site_general", "sex"], 10)

    with open("metadata.dat") as f:
        metadata = json.load(f)

    # df_train = datasets.ImageFolder("trainnew", transform=transform_train)
    # df_val = datasets.ImageFolder("valnew", transform=transform_test)
    df_train = metadata_functions.ImageFolderMetadata("trainnew", metadata, transform=transform_train)
    df_val = metadata_functions.ImageFolderMetadata("valnew", metadata, transform=transform_test)
    corrections = ImageCorrections()
    weights_train_long, weights_train = weights(df_train)

    if what == "train":

        # df_train0 = datasets.ImageFolder("trainnew", transform=transform_pre)
        # dataloader_preproc = DataLoader(df_train0, batch_size=20)
        # mean, sd = mean_sd(dataloader_preproc)
        # np.savetxt("mean.txt", mean.numpy())
        # np.savetxt("sd.txt", sd.numpy())
        dataloader_train = DataLoader(df_train, batch_size=22, pin_memory=True,
                                      shuffle=True,
                                      num_workers=10)

        dataloader_val = DataLoader(df_val, batch_size=22, pin_memory=True, num_workers=10)
        # dataloader_test = DataLoader(df_test, batch_size=20, pin_memory=True, num_workers=10)
        image_shower(dataloader_train)
        # model = CNN().to(device)
        model2 = EfficientNet.from_pretrained('efficientnet-b3', num_classes=50).to(device)
        # model3 = Metadata().to(device)
        model_withmeta = EfficientnetMetadata(model2).to(device)
        train_model(dataloader_train, dataloader_val, model_withmeta, is_metadata=True)
    elif what == "test":
        df_test = metadata_functions.ImageFolderMetadata("testnew", metadata=metadata, transform=transform_test)
        dataloader_test = DataLoader(df_test, batch_size=16, pin_memory=True, num_workers=8)
        model_own = torch.load("models/model421_iteration30.pt", map_location=device)
        test_model(dataloader_test, model_own, is_metadata=True)
    else:
        model_own = torch.load("models/model421_iteration30.pt", map_location=device)
        model2 = CNN().to(device)
        labels = []
        images = []
        filenames = []
        metadata_for_test = []
        for path, dirs, files in os.walk("C:/Users/pmarc/PycharmProjects/AI2/melanoma/testnew/"):
            for file in files:
                if path.split("/")[-1] == "melanoma":
                    labels.append(0)
                elif path.split("/")[-1] == "naevus":
                    labels.append(1)
                elif path.split("/")[-1] == "other":
                    labels.append(2)
                else:
                    print(path)
                filenames.append(path + "/" + file)

        choice = np.random.randint(0, len(labels), size=10)
        newlabels = []
        for i in choice:
            images.append(plt.imread(filenames[i]))
            newlabels.append(labels[i])
            metadata_for_test.append(torch.tensor(metadata[filenames[i].split("/")[-1].split(".")[0]]))

        # images.append(corrections.shades_of_gry(corrections.gamma_correction(plt.imread(filenames[0])))

        hooks = Hooks(model_own)

        hooks.saliency(images[0], torch.tensor([newlabels[0]], dtype=torch.long),
                       metadata=metadata_for_test[0])

        test_images(images, model_own.eval(), labels=newlabels, metadata=metadata_for_test)
