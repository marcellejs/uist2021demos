from tensorflow import keras
from tensorflow.keras import layers
from utils import load_datasets
from marcellepy import MarcelleCallback


class VGGlike:
    def __init__(self, img_shape, batch_size, learning_rate, epochs):
        self.img_shape = img_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["accuracy"]

    def conv(self, x, n_filters):
        x = layers.Conv2D(filters=n_filters, kernel_size=3, strides=2, padding="same")(
            x
        )
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    def build_model(self, input_shape, n_classes):
        inputs = layers.Input(shape=input_shape)
        x = self.conv(inputs, n_filters=256)
        x = self.conv(x, n_filters=128)
        x = self.conv(x, n_filters=128)
        x = self.conv(x, n_filters=64)
        x = layers.Flatten()(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="VGGlike")
        return model

    def train(self, train_ds, val_ds, class_names):
        model = self.build_model(input_shape=self.img_shape + (3,), n_classes=7)
        model.summary()

        # compile
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        # train callbacks:
        callbacks = [
            MarcelleCallback(
                model_checkpoint_freq=1,
                disk_save_formats=["h5", "tfjs"],
                remote_save_format="tfjs",
                run_params={
                    "model": "VGG16",
                    "img_shape": self.img_shape,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "class_names": class_names,
                },
            )
        ]

        model.fit(
            train_ds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_ds,
            verbose=1,
            callbacks=callbacks,
        )


if __name__ == "__main__":
    TRAIN_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000/processed/train_dir"
    IMG_SHAPE = (64, 64)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    train_ds, val_ds, class_names = load_datasets(
        TRAIN_DIR, BATCH_SIZE, IMG_SHAPE, preview=False
    )
    classifier = VGGlike(
        img_shape=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )
    classifier.train(train_ds, val_ds, class_names)
