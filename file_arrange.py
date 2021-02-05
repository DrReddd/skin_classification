import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


class DataManipulations:
    def __init__(self, data):
        self.combinations = {}
        self.unique_combination_counter = 0
        self.data = data

    def stratify_combinations_row(self, row, *args):
        combination = []
        for group in args:
            combination.append(row[group])
        combination = tuple(combination)
        if self.combinations.get(combination) is None:
            self.combinations[combination] = self.unique_combination_counter
            self.unique_combination_counter += 1
        return self.combinations[combination]

    def stratify_combinations(self, groups, min_num_ingroup):
        indecies = self.data.apply(self.stratify_combinations_row, axis=1, args=groups)
        return self.blend_groups(indecies, min_num_ingroup)

    @staticmethod
    def blend_groups(indecies: pd.Series, min_num_ingroup):
        indecies_numpy = indecies.to_numpy()
        for i in indecies.unique():
            if indecies_numpy[np.where(indecies_numpy == i)].shape[0] < min_num_ingroup:
                indecies.iloc[np.where(indecies_numpy == i)] = 0
        return indecies

    def unique(self, col):
        """returns indecies of elements which are unique in the series (only 1 occurence)"""
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


def image_label_dic_row(metadata_row, img_col, label_cols):
    for label in label_cols:
        if metadata_row[label] is True:
            return metadata_row[img_col]+".jpg", label


def image_label_dic(metadata, img_col, label_cols):
    pairs_series = metadata.apply(image_label_dic_row, axis=1, args=[img_col, label_cols])
    return dict([(img, label) for img, label in pairs_series.values])


def rearrange_files(path, train, val, test):
    """Rearrange files in my file system"""
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
unique_idc = df2_manipulate.unique("lesion_id")
unique_data = df2.iloc[unique_idc, :].copy()
only_for_training_data = df2.iloc[~df2.index.isin(unique_idc), :].copy()
new_group_indx = df2_manipulate.blend_groups(unique_data["group"], 10)
unique_data["group"] = new_group_indx

trainX, testX = train_test_split(unique_data, test_size=0.8, shuffle=True, stratify=unique_data["group"])

testX, valX = train_test_split(testX, test_size=0.3, shuffle=True, stratify=testX["group"])

trainX = pd.concat([only_for_training_data, trainX])

fig, axs = plt.subplots(1, 3)
axs[0].hist(trainX["group"], bins=70)
axs[1].hist(testX["group"], bins=70)
axs[2].hist(valX["group"], bins=70)
plt.show()


train_to_label = image_label_dic(trainX, "image", ("melanoma", "naevus", "other"))
val_to_label = image_label_dic(valX, "image", ("melanoma", "naevus", "other"))
test_to_label = image_label_dic(testX, "image", ("melanoma", "naevus", "other"))

rearrange_files("data/ISIC_2019_Training_Input/ISIC_2019_Training_Input/", train_to_label, val_to_label, test_to_label)
