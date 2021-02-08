import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple


class DataManipulations:
    """
    Contains some functions to manipulate metadata for the melanoma dataset
    """
    def __init__(self, data: pd.DataFrame):
        """

        :param data: dataframe
        """
        self.combinations = {}  # for applying stratification grouping to whole dataframe
        self.unique_combination_counter = 0  # for applying stratification grouping to whole dataframe
        self.data = data

    def stratify_combinations_row(self, row: pd.Series, *args) -> int:
        """
        Grouping metadata for stratified sampling.
        :param row: row from dataframe
        :param args: classes for grouping (column names)
        :return:
        """
        combination = []
        for group in args:
            combination.append(row[group])
        combination = tuple(combination)
        if self.combinations.get(combination) is None:
            self.combinations[combination] = self.unique_combination_counter
            self.unique_combination_counter += 1
        return self.combinations[combination]

    def stratify_combinations(self, groups: List[str], min_num_ingroup: int) -> pd.Series:
        """

        :param groups: column names for grouping
        :param min_num_ingroup: minimal number of members (rows) in a class
        :return: a column (series) of classes to copy into the original dataframe
        """
        indecies = self.data.apply(self.stratify_combinations_row, axis=1, args=groups)
        return self.blend_groups(indecies, min_num_ingroup)

    @staticmethod
    def blend_groups(indecies: pd.Series, min_num_ingroup: int) -> pd.Series:
        """
        Blends all classes with members less, then mi_num_ingroup into class 0
        :param indecies: series of classes
        :param min_num_ingroup: minimal number of objects (rows) in a class
        :return: series of classes, where every class has at least min_num_ingroup member
        """
        indecies_numpy = indecies.to_numpy()
        for i in indecies.unique():
            if indecies_numpy[np.where(indecies_numpy == i)].shape[0] < min_num_ingroup:
                indecies.iloc[np.where(indecies_numpy == i)] = 0
        return indecies

    def unique(self, col: str) -> List[int]:
        """
        Returns indecies of elements which are unique in the series (only 1 occurence)
        :param col: pd series
        :return: list of indexes
        """
        uniq = {}
        idxs = []
        series = self.data[col]
        for element in series:
            if uniq.get(element) is not None:
                uniq[element] = False
            else:
                uniq[element] = True
        for idx, element in enumerate(series):
            if uniq[element] is True:
                idxs.append(idx)

        return idxs


def image_label_dic_row(metadata_row: pd.Series, img_col: str, label_cols: List[str]) -> Tuple[str]:
    """
    Labels images with appropriate labels based on metadata, works on 1 row of metadata
    :param metadata_row: a row from a dataframe containing image ID (filename), and label (in a wide format)
    :param img_col: column (name) corresponding to image ID
    :param label_cols: boolean valued columns (names) corresponding to labels
    :return: tuple containing filename, label
    """
    for label in label_cols:
        if metadata_row[label] is True:
            return metadata_row[img_col] + ".jpg", label


def image_label_dic(metadata, img_col, label_cols) -> Dict[str, str]:
    """
    Applies image_label_dic_row() to whole dataframe
    :return dictionary, key is filename, value is label
    """
    pairs_series = metadata.apply(image_label_dic_row, axis=1, args=[img_col, label_cols])
    return dict([(img, label) for img, label in pairs_series.values])


def rearrange_files(path: str, train: Dict[str, str], val: Dict[str, str], test: Dict[str, str]) -> None:
    """
    Rearranges files into training, validation, test folders
    :param path: folder where images are originally placed
    :param train: dictionary containing the names of the training images, key is the name of the image, value is the label
    :param val: dictionary containing the names of the validation images, key is the name of the image, value is the label
    :param test: dictionary containing the names of the test images, key is the name of the image, value is the label
    :return: None
    """
    print(set(train.values()))
    for value in set(train.values()):
        try:
            os.makedirs(f"train/{value}")
            os.makedirs(f"test/{value}")
            os.makedirs(f"val/{value}")
        except FileExistsError:
            continue
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if train.get(filename) is not None:
                os.replace(dirpath + "/" + filename,
                           f"C:/Users/pmarc/PycharmProjects/AI2/melanoma/train/{train[filename]}/{filename}")
            elif test.get(filename) is not None:
                os.replace(dirpath + "/" + filename,
                           f"C:/Users/pmarc/PycharmProjects/AI2/melanoma/test/{test[filename]}/{filename}")
            elif val.get(filename) is not None:
                os.replace(dirpath + "/" + filename,
                           f"C:/Users/pmarc/PycharmProjects/AI2/melanoma/val/{val[filename]}/{filename}")


def arrange_back(newpath: str, train_val_test: List[str]) -> None:
    """
    Arranges files back into one directory
    :param newpath: a folder to put directories
    :param train_val_test: folders to copy from (training, validation, test folders)
    :return: None
    """
    for folder in train_val_test:
        for path, dirs, files in os.walk(folder):
            for file in files:
                os.replace(path + "/" + file,
                           f"{newpath}/{file}")


ground_truth1 = pd.read_csv("HAM10000_metadata.csv")
pics1 = "C:/Users/pmarc/PycharmProjects/AI2/melanoma/imgs/"
ground_truth2 = pd.read_csv("data/ISIC_2019_Training_GroundTruth.csv")
pics2 = "data/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"
tumor_names = {"akiec": "Act. Keratoses, intraep. carc.",
               "nv": "melanocytic nevi",
               "bkl": "bening keratosis-like lesions",
               "bcc": "basal cell carc.",
               "vasc": "vascular lesions",
               "mel": "melanoma",
               "df": "dermatofibroma"}

ground_truth1["melanoma"] = np.where(ground_truth1["dx"] == "mel", True, False)
ground_truth1["naevus"] = np.where(ground_truth1["dx"] == "nv", True, False)
ground_truth1["other"] = np.where((ground_truth1["dx"] != "mel") & (ground_truth1["dx"] != "nv"), True, False)
df1 = ground_truth1.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9]]

ground_truth2.drop((ground_truth2["UNK"] == 1).index)
ground_truth2["melanoma"] = np.where(ground_truth2["MEL"] == 1, True, False)
ground_truth2["naevus"] = np.where(ground_truth2["NV"] == 1, True, False)
ground_truth2["other"] = np.where((ground_truth2["MEL"] != 1) & (ground_truth2["NV"] != 1), True, False)
df2 = ground_truth2.iloc[:, [0, 10, 11, 12]]

metadata_df2 = pd.read_csv("data/ISIC_2019_Training_Metadata.csv")
df2 = pd.merge(df2, metadata_df2, on="image")
joint_df = pd.merge(df1, df2, left_on="image_id", right_on="image", how="outer")
print(joint_df.equals(df2))

df2_manipulate = DataManipulations(df2)
group_col = df2_manipulate.stratify_combinations(groups=("melanoma", "naevus", "other", "sex"),
                                                 min_num_ingroup=10)
df2["group"] = group_col

# unique_idc = df2_manipulate.unique("lesion_id")
# unique_data = df2.iloc[unique_idc, :].copy()
# only_for_training_data = df2.iloc[~df2.index.isin(unique_idc), :].copy()
# new_group_indx = df2_manipulate.blend_groups(unique_data["group"], 10)
# unique_data["group"] = new_group_indx

trainX, testX = train_test_split(df2, test_size=0.2, shuffle=True, stratify=df2["group"])

trainX, valX = train_test_split(trainX, test_size=0.1, shuffle=True, stratify=trainX["group"])

# trainX = pd.concat([only_for_training_data, trainX])

fig, axs = plt.subplots(1, 3)
axs[0].hist(trainX["group"], bins=70)
axs[1].hist(testX["group"], bins=70)
axs[2].hist(valX["group"], bins=70)
plt.show()

train_to_label = image_label_dic(trainX, "image", ("melanoma", "naevus", "other"))
val_to_label = image_label_dic(valX, "image", ("melanoma", "naevus", "other"))
test_to_label = image_label_dic(testX, "image", ("melanoma", "naevus", "other"))

rearrange_files("imgs/", train_to_label, val_to_label, test_to_label)
