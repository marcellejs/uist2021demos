import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, VGG16, ResNet50V2
from utils import load_datasets
from marcelle import Writer


def arrayize(x):
    return x if isinstance(x, list) else [x]


def augment_training_set(train_ds):
    img_augmentation_layers = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
            keras.layers.experimental.preprocessing.RandomFlip(),
            keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
        ]
    )
    return train_ds.map(
        lambda x, y: (img_augmentation_layers(x, training=True), y),
    )


class Classifier:
    def __init__(self, params):
        default_params = {
            "imagenet_weights": True,
            "data_augmentation": False,
            "fine_tune_layers": 0,
        }
        self.params = {**default_params, **params}
        self.input_shape = self.params["img_shape"] + (3,)
        self.num_classes = len(self.params["labels"])
        self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        if self.params["base_model"] == "efficientnet":
            x = keras.applications.efficientnet.preprocess_input(inputs)
            self.base_model = EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet" if self.params["imagenet_weights"] else None,
            )
        elif self.params["base_model"] == "mobilenet":
            x = keras.applications.mobilenet_v2.preprocess_input(inputs)
            self.base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet" if self.params["imagenet_weights"] else None,
            )
        elif self.params["base_model"] == "vgg16":
            x = keras.applications.vgg16.preprocess_input(inputs)
            self.base_model = VGG16(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet" if self.params["imagenet_weights"] else None,
            )
        elif self.params["base_model"] == "resnet50v2":
            x = keras.applications.resnet_v2.preprocess_input(inputs)
            self.base_model = ResNet50V2(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet" if self.params["imagenet_weights"] else None,
            )
        else:
            raise NotImplementedError("This model is not yet supported")

        # Freeze the pretrained weights
        self.base_model.trainable = False

        x = self.base_model(x)

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)

        # Compile
        self.model = keras.Model(
            inputs, outputs, name="%s_classifier" % self.params["base_model"]
        )

    def train(self, train_ds, val_ds):
        self.writer = Writer(
            disk_save_format="saved_model",
            remote_save_format="tfjs",
        )

        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.train_loss_metric = tf.keras.metrics.Mean(name="loss")
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        strategies = self.params["strategy"].split("+")
        all_epochs = arrayize(self.params["epochs"])
        all_lr = arrayize(self.params["learning_rate"])
        all_aug = arrayize(self.params["data_augmentation"])
        all_ftlayers = arrayize(self.params["fine_tune_layers"])

        self.total_epochs = int(np.sum(all_epochs))

        self.writer.create_run(self.model, self.params, self.loss.name)
        self.writer.train_begin(self.total_epochs)

        self.val_ds = val_ds

        initial_epoch = 0
        for i, strategy in enumerate(strategies):
            self.train_ds = augment_training_set(train_ds) if all_aug[i] else train_ds
            if strategy == "transfer":
                self.transfer(
                    epochs=all_epochs[i],
                    learning_rate=all_lr[i],
                    initial_epoch=initial_epoch,
                )
            elif strategy == "finetune":
                self.finetune(
                    epochs=all_epochs[i],
                    learning_rate=all_lr[i],
                    fine_tune_layers=all_ftlayers[i],
                    initial_epoch=initial_epoch,
                )
            elif strategy == "end2end":
                self.end2end(
                    epochs=all_epochs[i],
                    learning_rate=all_lr[i],
                    initial_epoch=initial_epoch,
                )
            initial_epoch += all_epochs[i]
        self.writer.train_end(save_checkpoint=True)

    def transfer(self, epochs, learning_rate, initial_epoch=0):
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        print(self.model.summary())

        self.train_phase(epochs, initial_epoch)

    def finetune(self, epochs, learning_rate, fine_tune_layers, initial_epoch=0):
        self.model.trainable = True
        self.base_model.trainable = True

        num_layers = len(self.base_model.layers)
        # Freeze all the layers before the `fine_tune_at` layer
        last_frozen_layer = num_layers - fine_tune_layers

        print("Number of layers in the base model: ", num_layers)
        print("last_frozen_layer", last_frozen_layer)

        for layer in self.base_model.layers[:last_frozen_layer]:
            layer.trainable = False

        self.optimizer = keras.optimizers.RMSprop(lr=learning_rate)

        print(self.model.summary())

        self.train_phase(epochs, initial_epoch)

    def end2end(self, epochs, learning_rate, initial_epoch=0):
        self.model.trainable = True
        self.base_model.trainable = True

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        print(self.model.summary())

        self.train_phase(epochs, initial_epoch)

    def train_phase(self, epochs, initial_epoch):
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = self.loss(y, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            self.train_acc_metric.update_state(y, logits)
            self.train_loss_metric(loss_value)
            return loss_value

        @tf.function
        def test_step(x, y):
            val_logits = self.model(x, training=False)
            val_loss_value = self.loss(y, val_logits)
            self.val_acc_metric.update_state(y, val_logits)
            return val_loss_value

        for epoch in range(initial_epoch, initial_epoch + epochs):
            print("\nEpoch {}/{}".format(epoch + 1, self.total_epochs))
            pb_i = keras.utils.Progbar(
                len(self.train_ds),
                verbose=1,
                stateful_metrics=["loss", "accuracy"],
                unit_name="step",
            )

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                loss_value = train_step(x_batch_train, y_batch_train)
                values = [
                    ("loss", loss_value.numpy()),
                    ("accuracy", self.train_acc_metric.result().numpy()),
                ]
                pb_i.update(step, values=values)

            train_acc = self.train_acc_metric.result().numpy()
            train_loss = self.train_loss_metric.result().numpy()
            self.train_acc_metric.reset_states()
            self.train_loss_metric.reset_states()

            for x_batch_val, y_batch_val in self.val_ds:
                val_loss_value = test_step(x_batch_val, y_batch_val)
                self.val_loss_metric(val_loss_value)
            val_acc = self.val_acc_metric.result().numpy()
            val_loss = self.val_loss_metric.result().numpy()
            self.val_acc_metric.reset_states()
            self.val_loss_metric.reset_states()

            logs = {
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            pb_i.update(step + 1, values=list(logs.items()), finalize=True)
            self.writer.save_epoch(epoch, logs=logs, save_checkpoint=True)


def params_end2end(base):
    return {
        "img_shape": (224, 224),
        "batch_size": 32,
        "base_model": base,
        "imagenet_weights": False,
        "strategy": "end2end",
        "epochs": 10,
        "learning_rate": 1e-3,
        "data_augmentation": True,
    }


def params_transfer_finetune(base):
    return {
        "img_shape": (224, 224),
        "batch_size": 32,
        "base_model": base,
        "imagenet_weights": True,
        "strategy": "transfer+finetune",
        "epochs": [10, 10],
        "learning_rate": [1e-3, 1e-4],
        "data_augmentation": [False, False],
        "fine_tune_layers": [0, 30],
    }


if __name__ == "__main__":
    TRAIN_DIR = "/Users/jules/Documents/Research/Datasets/HAM10000/processed/train_dir"
    for PARAMS in [
        # params_end2end("mobilenet"),
        params_transfer_finetune("mobilenet"),
        # params_end2end("resnet50v2"),
        params_transfer_finetune("resnet50v2"),
        # params_end2end("efficientnet"),
        params_transfer_finetune("efficientnet"),
        # params_end2end("vgg16"),
        params_transfer_finetune("vgg16"),
    ]:
        TRAIN_DS, VAL_DS, LABELS = load_datasets(
            train_dir=TRAIN_DIR,
            batch_size=PARAMS["batch_size"],
            img_shape=PARAMS["img_shape"],
            preview=False,
        )
        PARAMS["labels"] = LABELS
        classifier = Classifier(PARAMS)
        classifier.train(TRAIN_DS, VAL_DS)
