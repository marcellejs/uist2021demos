import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


HAM10000_DATA_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000"
TARGET_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000/processed"
TRAIN_DIR = os.path.join(TARGET_DIR, "train_dir")
TEST_DIR = os.path.join(TARGET_DIR, "test_dir")

# note that we are not augmenting class 'nv'
class_list = ["mel", "bkl", "bcc", "akiec", "vasc", "df"]

for img_class in class_list:
    # We are creating temporary directories here because we delete these
    # directories later create a base dir
    aug_dir = "aug_dir"
    os.mkdir(aug_dir)
    # create a dir within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, "img_dir")
    os.mkdir(img_dir)

    # list all images in that directory
    img_list = os.listdir(f"{TRAIN_DIR}/{img_class}")
    print(f"{img_class} => Found {len(img_list)} images.")

    # Copy images from the class train dir to the img_dir e.g. class 'mel'
    for fname in img_list:
        # source path to image
        src = os.path.join(f"{TRAIN_DIR}/{img_class}", fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = f"{TRAIN_DIR}/{img_class}"

    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        # brightness_range=(0.9,1.1),
        fill_mode="nearest",
    )

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(
        path,
        save_to_dir=save_path,
        save_format="jpg",
        target_size=(224, 224),
        batch_size=batch_size,
    )

    # Generate the augmented images and add them to the training folders

    ###########

    num_aug_images_wanted = 6000  # total number of images we want to have in each class

    ###########

    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    # run the generator and create about 6000 augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree("aug_dir")
