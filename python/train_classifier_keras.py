from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from utils import load_datasets
from marcelle import KerasCallback


def arrayize(x):
    return x if isinstance(x, list) else [x]


class Classifier:
    def __init__(self, params):
        self.params = params
        self.input_shape = self.params["img_shape"] + (3,)
        self.num_classes = len(self.params["labels"])
        self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        self.base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
        )

        # Freeze the pretrained weights
        self.base_model.trainable = False

        x = self.base_model(inputs)

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)

        # Compile
        self.model = keras.Model(inputs, outputs, name="efficientnet_keras_classifier")

    def train(self, train_ds, val_ds):
        self.callbacks = [
            KerasCallback(
                model_checkpoint_freq=1,
                disk_save_format="h5",
                remote_save_format="tfjs",
                run_params=self.params,
            ),
        ]

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(self.model.summary())

        self.model.fit(
            train_ds,
            epochs=self.params["epochs"],
            callbacks=self.callbacks,
            validation_data=val_ds,
            verbose=1,
        )


if __name__ == "__main__":
    TRAIN_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000/processed/train_dir"
    PARAMS = {
        "img_shape": (224, 224),
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 1e-3,
    }
    train_ds, val_ds, labels = load_datasets(
        train_dir=TRAIN_DIR,
        batch_size=PARAMS["batch_size"],
        img_shape=PARAMS["img_shape"],
        preview=True,
    )
    PARAMS["labels"] = labels
    classifier = Classifier(PARAMS)
    classifier.train(train_ds, val_ds)
