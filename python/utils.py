from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_datasets(train_dir, batch_size, img_shape, preview=False):
    train_ds = image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=img_shape,
        shuffle=True,
        seed=1337,
        validation_split=0.2,
        subset="training",
    )
    val_ds = image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=img_shape,
        shuffle=True,
        seed=1337,
        validation_split=0.2,
        subset="validation",
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    if preview:
        return train_ds.take(2), val_ds.take(2), class_names

    return train_ds, val_ds, class_names
