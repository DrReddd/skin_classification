import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import pandas as pd
import json
import numpy as np
from typing import List, Dict, Optional, Callable, Any, Tuple
import numbers

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def encode_csv(csv_file: pd.DataFrame, cols_to_encode: List[str], one_hot_max: int) -> List[List]:
    """
    Encodes metadata columns in a table format to one-hot vectors or in case of too much unique values, a numeric
    vector.
    :param csv_file: metadata
    :param cols_to_encode: columns containing relevant metadata
    :param one_hot_max: maximal unique values in a column to encode it into one-hot format
    :return: encoded metadata
    """
    all_encode = list()
    for column in cols_to_encode:
        encoded_col = list()
        if csv_file[column].nunique() < one_hot_max:
            unique_vals = csv_file[column].dropna().unique()
            unique_vals = np.concatenate((unique_vals, np.array(["no data"])))
            one_hot_encode = np.zeros(unique_vals.shape)
            for value in csv_file[column]:
                if pd.isnull(value):
                    one_index = np.where(unique_vals == "no data")
                else:
                    one_index = np.where(unique_vals == value)
                one_hot_encode[one_index] = 1
                encoded_col.append(one_hot_encode.copy())
                one_hot_encode[one_index] = 0
        else:
            for value in csv_file[column]:
                assert isinstance(value, numbers.Number)
                if pd.isnull(value):
                    encoded_col.append([-40])
                else:
                    encoded_col.append([value])
        all_encode.append(encoded_col)
    return all_encode


def jsonify(names: List[str], csv_file: pd.DataFrame, cols_to_encode: List[str], one_hot_max: int) -> None:
    """
    Saves metadata into json format.
    :param names: Image names
    :param csv_file: dataframe containing the metadata of the images in the same order as the images are in "names"
    :param cols_to_encode: columns containing relevant metadata
    :param one_hot_max: maximal unique values in a column to encode it into one-hot format
    :return: None
    """
    encoding = encode_csv(csv_file, cols_to_encode, one_hot_max)
    data_for_json = {}
    for idx, image_name in enumerate(names):
        img_metadata = list()
        for vectors in encoding:
            img_metadata.extend(vectors[idx])
        data_for_json[image_name] = img_metadata
    with open("metadata.dat", "w") as f:
        json.dump(data_for_json, f)


class ImageFolderMetadata(datasets.DatasetFolder):
    """
    A custom dataloader, which returns metadata alongside image and label, if it exists.
    """
    def __init__(
            self,
            root: str,
            metadata: Optional[Dict[str, np.ndarray]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.metadata = metadata
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Loads the appropriate data for the pytorch framework.
        :param index: index of the image
        :return: image, class, and metadata if it exists
        """

        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img_name = path.split("\\")[-1].split(".")[0]

        if self.metadata is not None:
            return sample, target, torch.tensor(self.metadata[img_name])
        else:
            return sample, target
