import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


HAM10000_DATA_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000"
TARGET_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000/processed"
TRAIN_DIR = os.path.join(TARGET_DIR, "train_dir")
TEST_DIR = os.path.join(TARGET_DIR, "test_dir")


def create_dirs():
    os.mkdir(TARGET_DIR)
    os.mkdir(TRAIN_DIR)
    os.mkdir(TEST_DIR)

    for label in ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]:
        os.mkdir(os.path.join(TRAIN_DIR, label))
        os.mkdir(os.path.join(TEST_DIR, label))


def copy_images():
    meta = pd.read_csv(os.path.join(HAM10000_DATA_DIR, "HAM10000_metadata.csv"))

    print(meta.head())

    # now we create a val set using df because we are sure that none of these images
    # have augmented duplicates in the train set
    df_train, df_val = train_test_split(
        meta, test_size=0.1, random_state=101, stratify=meta["dx"]
    )

    print(df_train.head())
    print(df_val.head())
    print(df_train.shape, df_val.shape)

    # Set the image_id as the index in df_data
    # df_data.set_index('image_id', inplace=True)

    # # Get a list of images in each of the two folders
    folder_1 = os.listdir(os.path.join(HAM10000_DATA_DIR, "ham10000_images_part_1"))
    folder_2 = os.listdir(os.path.join(HAM10000_DATA_DIR, "ham10000_images_part_2"))

    # Transfer the train images
    for index, row in df_train.iterrows():
        fname = row["image_id"] + ".jpg"
        label = row["dx"]

        if fname in folder_1:
            src = os.path.join(HAM10000_DATA_DIR, "ham10000_images_part_1", fname)
            dst = os.path.join(TRAIN_DIR, label, fname)
            shutil.copyfile(src, dst)

        if fname in folder_2:
            src = os.path.join(HAM10000_DATA_DIR, "ham10000_images_part_2", fname)
            dst = os.path.join(TRAIN_DIR, label, fname)
            shutil.copyfile(src, dst)

    # Transfer the val images
    for index, row in df_val.iterrows():
        fname = row["image_id"] + ".jpg"
        label = row["dx"]

        if fname in folder_1:
            src = os.path.join(HAM10000_DATA_DIR, "ham10000_images_part_1", fname)
            dst = os.path.join(TEST_DIR, label, fname)
            shutil.copyfile(src, dst)

        if fname in folder_2:
            src = os.path.join(HAM10000_DATA_DIR, "ham10000_images_part_2", fname)
            dst = os.path.join(TEST_DIR, label, fname)
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    create_dirs()
    copy_images()
